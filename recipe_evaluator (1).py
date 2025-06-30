class RecipeEvaluator:
    """
    Evaluates generated recipes using the DeepInfra API.
    The class sends prompts to an LLM for structured scoring and extracts evaluation results.
    """

    def __init__(self, deepinfra_api_key):
        # Store the API key for authenticating with DeepInfra's endpoint
        self._deepinfra_api_key = deepinfra_api_key

    async def evaluate_recipe_with_llm(self, recipe, island_id, model_name):
        """
        Sends a structured prompt to DeepInfra's LLM to critically evaluate a recipe.
        Returns extracted creativity and quality scores in a consistent JSON format.

        Parameters:
        - recipe (str): The generated recipe text.
        - island_id (int): Identifier for the originating island.
        - model_name (str): Either 'good' or any fallback model string.
        """

        # Translate user-friendly model name to actual API model identifier
        if model_name == "good":
            model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        else:
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        # Construct a strict prompt that forces the LLM to return scores in structured JSON
        prompt = f"""
        You are a highly critical judge in the Pillsbury Bake-Off, and your standards are exceptionally high. Your role is to rate recipes with great scrutiny, focusing on both the technical and emotional aspects. Please analyze the following recipe in the <recipe> tag according to the instructions in the <instructions> tag and provide your response in the JSON format specified in the <output_format> tag.

        <recipe>
        {recipe}
        </recipe>

        <instructions>
        For each of the following dimensions, **first provide a detailed qualitative assessment** followed by a score (1 to 5). **It is mandatory that you provide a score immediately after each qualitative assessment for all dimensions.** You **must** provide a score for each dimension, even if the qualitative assessment is brief.

        ### Recipe Judging:
        1) **Taste**: Evaluate how well-balanced and pleasing the flavors are in the dish, considering aspects like seasoning, texture, and overall flavor profile. (1 low - 5 high)
        2) **Appearance**: Rate how visually appealing the dish is, considering factors like color contrast, plating, and presentation. (1 low - 5 high)
        3) **Creativity**: Assess the innovation and originality of the recipe, considering ingredient combinations, cooking techniques, and presentation. (1 low - 5 high)
        4) **Crowd Appeal**: Determine how likely the dish is to be enjoyed by a wide range of people, considering its familiarity, comfort, and versatility. (1 low - 5 high)

        ### Story Judging:
        1) **How the recipe ties to the story**: Does the recipe reflect the story behind it, making the dish feel authentic to the narrative? (1 low - 5 high)
        2) **How the story brings to life a family value, tradition, or memory**: Does the story evoke emotions tied to family or tradition, adding depth to the recipe? (1 low - 5 high)
        3) **Demonstration of Passion**: Does the story showcase a deep, genuine emotional connection to the recipe? Does it convey the chef’s personal love for the dish, the culinary tradition, or cooking in general? **This score is absolutely crucial** and must be clearly articulated in your assessment. (1 low - 5 high)

        ### Overall score:
        1) **How would you rate the overall recipe with respect to alle these dimensions?** (1 low - 5 high)

        **Important Notes:**
        - **You MUST provide scores immediately after each quantitative assessment: taste, appearance, creativity, crowd_appeal, recipe_ties_story, story_brings_to_life, passion and overall. You must rate all these assessments
        Provide your response **only** in this strict JSON format, without any extra commentary.
        </instructions>

        <output_format>
        {{
            "taste_quality_assess": <your_assessment>,
            "taste": <score between 1-5>,
            "appearance_quality_assess": <your_assessment>,
            "appearance": <score between 1-5>,
            "creativity_quality_assess": <your_assessment>,
            "creativity": <score between 1-5>,
            "crowd_appeal_quality_assess": <your_assessment>,
            "crowd_appeal": <score between 1-5>,
            "recipe_ties_story_quality_assess": <your_assessment>,
            "recipe_ties_story": <score between 1-5>,
            "story_brings_to_life_quality_assess": <your_assessment>,
            "story_brings_to_life": <score between 1-5>,
            "passion_quality_assess": <your_assessment>,
            "passion": <score between 1-5>,
            "overal_qualitiy_assess": <your_assessment>,
            "overall": <score between 1-5>
        }}
        </output_format>
        """

        # Headers and URL for DeepInfra chat completions endpoint
        headers = {"Authorization": f"Bearer {self._deepinfra_api_key}", "Content-Type": "application/json"}
        url = f"https://api.deepinfra.com/v1/openai/chat/completions"  # Updated URL

        # JSON payload to send to the API
        data = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.25,
            "max_tokens": 512
        }
        # Make asynchronous request and extract the response
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    response_json = await response.json()
                    logging.info(f"Received response: {json.dumps(response_json, indent=2)}")

                    if response_json and "choices" in response_json:
                        generated_text = response_json["choices"][0]["message"]["content"].strip()
                        extract_scores = await self.extract_scores_from_response(generated_text, recipe, island_id)  # Use 'self' to call the method
                        return extract_scores
                    else:
                        logging.error("No valid response from DeepInfra API.")
                        return {}
        except Exception as e:
            logging.error(f"Error calling DeepInfra API: {e}")
            return {}


    async def extract_scores_from_response(self, response_text, recipe, island_id):
        """
        Parses the response text to extract numeric scores for various recipe and story dimensions.
        Implements multiple fallback methods: JSON parsing, regex extraction, and loose matching.
        Returns:
        - dict with raw scores, a weighted overall score, and island identifier.
        """
        score_keys = [
                "taste", "appearance", "creativity", "crowd_appeal",
                "recipe_ties_story", "story_brings_to_life", "passion"
            ]
        scores = {}
        recipe_info = {}
        # Attempt to parse strict JSON block
        json_match = re.search(r"\{[\s\S]*?\}", response_text)
        if json_match:
            json_text = json_match.group(0).strip()
            try:
                response_json = json.loads(json_text)
                for key in score_keys:
                    if key in response_json and isinstance(response_json[key], (int, float)) and 1 <= response_json[key] <= 5:
                        scores[key] = float(response_json[key])

            except json.JSONDecodeError:
                pass  # Continue with other strategies

        # Regex fallback for key-value pattern if JSON fails
        if len(scores) < 7:
            for key in score_keys:
                pattern = rf'"?\b{key}\b"?\s*[:=]\s*"?([1-5](?:\.\d+)?)"?'
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match and key not in scores:
                    try:
                        scores[key] = float(match.group(1))
                    except ValueError:
                        pass

        # Last-resort fallback: loosely match score by keyword and nearby number
        if len(scores) < 7:
            for key in score_keys:
                pattern = rf"{key}.*?\b([1-5])\b"
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match and key not in scores:
                    try:
                        scores[key] = float(match.group(1))
                    except ValueError:
                        pass

        # Ensure all expected keys exist (default to 0)
        scores_per_test = {key: float(scores.get(key, 0)) for key in score_keys}

        # Custom weighting scheme for final score calculation
        weights = {
                "taste": 0.175, "appearance": 0.175, "creativity": 0.175, "crowd_appeal": 0.175,
                "recipe_ties_story": 0.12, "story_brings_to_life": 0.12, "passion": 0.06
            }

        # Compute weighted average score
        score = round(sum(scores_per_test[key] * weights[key] for key in weights), 2)

        return {
            "scores": [scores.get(key, None) for key in score_keys],
            "weighted_score": score,
            "island_id": island_id
        }

    async def extract_recipe_details(self, response_text):
        """
        Parses structured recipe details from a raw LLM response string.

        Extracts:
        - recipe_idea
        - essay
        - recipe_name
        - ingredients (list)
        - instructions (list)
        """
        recipe_details = {}

        # Patterns to extract specific fields (handles both strings and arrays)
        patterns = {
            "recipe_idea": r'"recipe_idea"\s*[:=]\s*(\[[^\]]*\]|"[^"]*")',
            "essay": r'"essay"\s*[:=]\s*(\[[^\]]*\]|"[^"]*")',
            "recipe_name": r'"recipe_name"\s*[:=]\s*(\[[^\]]*\]|"[^"]*")',
            "ingredients": r'"ingredients"\s*[:=]\s*(\[[^\]]*\]|"[^"]*")',
            "instructions": r'"instructions"\s*[:=]\s*(\[[^\]]*\]|"[^"]*")'
        }

        # Extract and normalize each field
        for field, pattern in patterns.items():
            match = re.search(pattern, response_text)
            if match:
                value = match.group(1).strip()

                # Clean and split ingredients into a list
                if field == "ingredients":
                    if value.startswith("[") and value.endswith("]"):
                        recipe_details[field] = [ingredient.strip().strip('"') for ingredient in value[1:-1].split(',')]
                    else:
                        recipe_details[field] = [ingredient.strip().strip('"') for ingredient in value.split(',')]

                # Handle multi-line or array-like instructions
                elif field == "instructions":
                    if value.startswith("[") and value.endswith("]"):
                        recipe_details[field] = [instruction.strip().strip('"') for instruction in value[1:-1].split(',')]
                    else:
                        recipe_details[field] = [line.strip() for line in value.split("\\n") if line.strip()]

                # Basic string fields
                else:
                    recipe_details[field] = value

        return recipe_details
