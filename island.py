class Island:
    """A sub-population of the programs database, managing recipes and clusters."""

    # Class-level DataFrame to hold programs for all islands
    _programs = pd.DataFrame(columns=[
        "island_id", "recipe_idea", "essay", "recipe_name",
        "ingredients", "instructions", "taste", "appearance", "creativity",
        "crowd_appeal", "recipe_ties_story", "story_brings_to_life",
        "passion", "weighted_score", "better_format"
    ])

    def __init__(self, island_id: int, template: str, functions_per_prompt: int, deepinfra_api_key, evaluator: RecipeEvaluator) -> None:
        """
        Initialize an island object to generate and evaluate recipes.

        Args:
            island_id (int): Unique identifier for the island.
            template (str): Recipe template to guide recipe generation.
            functions_per_prompt (int): Number of functions per prompt to adjust.
            deepinfra_api_key (str): API key for DeepInfra.
            evaluator (RecipeEvaluator): Instance of RecipeEvaluator to assess the generated recipes.
        """
        self.island_id = island_id
        self.template = template
        self._functions_per_prompt = functions_per_prompt
        self._deepinfra_api_key = deepinfra_api_key
        self._best_program = None
        self._best_score = float('-inf')
        self._clusters = []
        self.evaluator = evaluator

    async def initialize_programs(self, seed, generator_temperature, model_name, allow_infinite_retries: bool = True):
        """
        Continuously attempt to generate a valid program (recipe) for this island until success or max retries.

        Args:
            seed (str): The seed text for recipe generation.
            generator_temperature (float): Temperature for recipe generation.
            model_name (str): Name of the model to be used for evaluation.
            allow_infinite_retries (bool): Flag to allow infinite retries or set a max retry limit.
        """
        max_total_attempts = 100  # Prevent infinite loop edge cases
        attempt = 0

        while True:
            attempt += 1
            logging.info(f"[Island {self.island_id}] Attempt {attempt} to generate a recipe.")

            program = await self.generate_recipe(seed, generator_temperature)
            if not program:
                logging.warning(f"[Island {self.island_id}] Failed to generate a recipe. Retrying...")
                await asyncio.sleep(random.uniform(0.5, 1.5))
                if not allow_infinite_retries and attempt >= max_total_attempts:
                    logging.error(f"[Island {self.island_id}] Max attempts reached. Giving up.")
                    break
                continue

            response = await self.evaluator.evaluate_recipe_with_llm(program, self.island_id, model_name)

            previous_program_count = len(Island._programs)
            await self.register_and_evaluate_program(program, self.island_id, response)

            if len(Island._programs) > previous_program_count:
                logging.info(f"[Island {self.island_id}] Successfully registered a valid program.")
                break  # Done
            else:
                logging.warning(f"[Island {self.island_id}] Program was invalid. Retrying...")
                await asyncio.sleep(random.uniform(0.5, 2.0))

            if not allow_infinite_retries and attempt >= max_total_attempts:
                logging.error(f"[Island {self.island_id}] Max attempts reached. Giving up.")
                break


    async def generate_recipe(self, seed, generator) -> str:
        """
        Generate a recipe using the given template and DeepInfra API.

        Args:
            seed (str): The seed text used for generation.
            generator (float): Temperature setting for recipe generation.

        Returns:
            str: The generated recipe text.
        """
        prompt = f"{self.template}"
        logging.info(f"Generating recipe with seed: {seed}")

        headers = {"Authorization": f"Bearer {self._deepinfra_api_key}", "Content-Type": "application/json"}
        url = "https://api.deepinfra.com/v1/openai/chat/completions"

        data = {
            "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": generator,
            "max_tokens": 512
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    response_json = await response.json()
                    if response_json and "choices" in response_json:
                        return response_json["choices"][0]["message"]["content"].strip()
                    else:
                        logging.error("Unexpected response format")
                        return ""
        except Exception as e:
            logging.error(f"Error calling DeepInfra API: {e}")
        return ""

    def calculate_island_score(self):
        """
        Calculate the average weighted score for all programs in the island.

        Returns:
            float: The average weighted score of all programs for the island.
        """
        island_programs = Island._programs[Island._programs["island_id"] == self.island_id]
        return island_programs["weighted_score"].mean() if not island_programs.empty else 0

    def rank_programs(self):
        """
        Rank programs by their weighted score using softmax.

        Returns:
            pd.DataFrame: A DataFrame of programs sorted by their softmax score.
        """
        island_programs = Island._programs[Island._programs["island_id"] == self.island_id].copy()

        if island_programs.empty:
            return pd.DataFrame()

        # Compute softmax scores
        scores = island_programs["weighted_score"].values
        island_programs["softmax_score"] = softmax(scores)

        # Return sorted DataFrame without modifying the original
        return island_programs.sort_values(by="softmax_score", ascending=False)

    async def register_and_evaluate_program(self, program, island_id, response):
        """
        Registers a program if it is valid and has a better score than the current best program.

        Args:
            program (str): The generated recipe to evaluate and register.
            island_id (int): Unique identifier for the island.
            response (dict): The response from the evaluator with scores and other details.
        """
        # Step 1: Extract recipe details
        recipe = await self.evaluator.extract_recipe_details(program)

        # Normalize and validate instructions
        raw_instructions = recipe.get("instructions")
        if isinstance(raw_instructions, list):
            instructions = " ".join([i.strip('"') for i in raw_instructions if isinstance(i, str)])
        elif isinstance(raw_instructions, str):
            instructions = raw_instructions.strip()
        else:
            instructions = ""

        # Normalize and validate ingredients
        raw_ingredients = recipe.get("ingredients")
        if isinstance(raw_ingredients, list):
            ingredients = [i.strip('"') for i in raw_ingredients if isinstance(i, str) and i.strip()]
        elif isinstance(raw_ingredients, str):
            # Assume it's a comma-separated string blob, split it
            ingredients = [i.strip() for i in raw_ingredients.split(",") if i.strip()]
        else:
            ingredients = []

        # Check required fields
        missing_fields = []
        if not isinstance(recipe.get("recipe_name"), str) or not recipe["recipe_name"].strip():
            missing_fields.append("recipe_name")
        if not instructions:
            missing_fields.append("instructions")
        if not ingredients:
            missing_fields.append("ingredients")

        if missing_fields:
            return

        # Step 3: Validate scores
        scores = response.get('scores', [None] * 7)
        if (
            not isinstance(scores, list) or len(scores) != 7 or
            any(s is None or math.isnan(s) for s in scores) or
            "weighted_score" not in response or
            response["weighted_score"] is None or math.isnan(response["weighted_score"])
        ):
            return

        # Unpack scores
        taste, appearance, creativity, crowd_appeal, recipe_ties_story, story_brings_to_life, passion = scores

        # Step 4: Format structured data
        formatted_data = {
            "island_id": response.get("island_id", island_id),
            "recipe_idea": recipe.get("recipe_idea", ""),
            "essay": recipe.get("essay", ""),
            "recipe_name": recipe["recipe_name"].strip(),
            "ingredients": ingredients,
            "instructions": instructions,
            "taste": taste,
            "appearance": appearance,
            "creativity": creativity,
            "crowd_appeal": crowd_appeal,
            "recipe_ties_story": recipe_ties_story,
            "story_brings_to_life": story_brings_to_life,
            "passion": passion,
            "weighted_score": response["weighted_score"],
            "formatted_recipe": ""
        }

        # Step 5: Append to class-wide programs DataFrame
        Island._programs = pd.concat(
            [Island._programs, pd.DataFrame([formatted_data])],
            ignore_index=True
        )

        # Step 6: Update best program if this one is better
        self._update_best_program(formatted_data)



    def cluster_programs(self):
        """
        Cluster programs within the island based on their weighted scores.

        This method groups programs into clusters, where each cluster represents
        a set of programs that are similar in terms of their weighted score.
        """
        island_programs = Island._programs[Island._programs["island_id"] == self.island_id]

        if island_programs.empty:
            return

        # Group the programs by their weighted score (or another feature of choice)
        for _, row in island_programs.iterrows():
            program = row.to_dict()  # Convert each row to a dictionary
            score = row['weighted_score']

            # Check if this program fits into an existing cluster or if a new cluster is needed
            added_to_cluster = False
            for cluster in self._clusters:
                # Here you could apply a more sophisticated clustering condition, e.g., a threshold score difference
                if abs(cluster.score - score) < 0.5:  # Example threshold for clustering
                    cluster.add_program(program, score)
                    added_to_cluster = True
                    break

            if not added_to_cluster:
                # Create a new cluster for this program
                new_cluster = Cluster(score, program)
                self._clusters.append(new_cluster)

    def get_best_program_from_clusters(self):
        """
        Get the best program across all clusters in the island.

        Returns:
            dict: The best program (dict) from all clusters.
        """
        best_program = None
        best_score = float('-inf')

        for cluster in self._clusters:
            cluster_best_program = cluster.get_best_program()
            cluster_best_score = cluster_best_program['score']

            if cluster_best_score > best_score:
                best_program = cluster_best_program
                best_score = cluster_best_score

        return best_program



    def _update_best_program(self, program_data):
        """
        Update the best program if the current one has a higher weighted score.

        Args:
            program_data (dict): The data for the program to be compared.
        """
        if program_data["weighted_score"] > self._best_score:
            self._best_program = program_data
            self._best_score = program_data["weighted_score"]

    def get_best_program(self):
        """
        Retrieve the best program for the island.

        Returns:
            dict: The best program for the island.
        """
        return self._best_program
