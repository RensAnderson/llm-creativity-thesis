async def evolve_function(previous_versions, deepinfra_api_key, template, generator_temperature) -> Dict[str, Any]:
    """
    Calls DeepInfra API asynchronously to evolve a recipe.

    This function prompts the LLM to generate a new, creative recipe that improves upon
    a list of prior versions, while adhering to a specific contest template.

    Args:
        previous_versions (List[dict]): A list of previous recipe dictionaries to improve upon.
        deepinfra_api_key (str): API key for authentication with DeepInfra.
        template (str): Contest template outlining requirements for the recipe.
        generator_temperature (float): Sampling temperature for creative diversity.

    Returns:
        Dict[str, Any]: A dictionary containing the evolved recipe fields (or empty dict on failure).
    """

    # Prompt instructing the LLM to generate a better version of existing recipes
    prompt = f"""
              You are a renowned chef participating in a high-profile international contest: The Pillsbury Bake-Off.
              Please create a new recipe according to the instructions in the <instructions> tag and provide your response in the JSON format as specified in the <output_format> tag.

              <instructions>
              You should create and return a **better, more creative, and different version** of the following recipe and essay: {previous_versions}. The new recipe must surpass the previous one in creativity, originality, and presentation, while still strictly adhering to all contest rules outlined in the {template}.
              Your answer must include, without exception, the following components:
              - Recipe Idea
              - Essay
              - Recipe Name
              - Ingredients (max 10, excluding pantry staples)
              - Instructions (clear, concise, and within 2,000 characters)

              For each component (Recipe Idea, Essay, Recipe Name, Ingredients, Instructions), you must provide your response immediately following the qualitative assessment.

              Please provide your response strictly in the following JSON format, without any extra commentary.
              IMPORTANT: Ensure your response **fully completes** the recipe and ends with }}.
              </instructions>

              <output>
              {{
                  "recipe_idea": <your_recipe_idea>,
                  "essay": <your_essay>,
                  "recipe_name": <your_recipe_name>,
                  "ingredients": <your_recipe_ingredients>,
                  "instructions": <your_recipe_instructions>
              }}
              </output>
              """

    headers = {
        "Authorization": f"Bearer {deepinfra_api_key}",
        "Content-Type": "application/json"
    }

    url = "https://api.deepinfra.com/v1/openai/chat/completions"

    data = {
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": generator_temperature,
        "max_tokens": 512
    }

    # Send request asynchronously
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            response_json = await response.json()

            if response_json and "choices" in response_json:
                generated_text = response_json["choices"][0]["message"]["content"].strip()
                if generated_text:
                    return generated_text  # Expecting a JSON string to be parsed later

    return {}  # Return empty on failure

async def fun_search_optimization(
    num_batches,
    recipes_per_batch,
    num_islands,
    generator_temperature,
    model_name,
    template,
    deepinfra_api_key
):
    """
    Orchestrates the FunSearch algorithm for creative recipe evolution.

    This method performs evolutionary optimization over multiple batches and islands,
    simulating parallel recipe development and improvement.

    Args:
        num_batches (int): Number of evolution rounds (batches).
        recipes_per_batch (int): Number of new recipes to generate per island per batch.
        num_islands (int): Number of independent islands to evolve separately.
        generator_temperature (float): Creativity level for recipe generation.
        model_name (str): Name of the LLM used for evaluation.
        template (str): Recipe template containing contest rules.
        deepinfra_api_key (str): API key to access the DeepInfra generation model.

    Returns:
        pd.DataFrame: A DataFrame of all evolved and scored recipes.
    """
    # Clear any existing program state
    Island._programs = pd.DataFrame(columns=Island._programs.columns)

    # Initialize program manager and islands
    programs_db = ProgramsDatabase(template, deepinfra_api_key)
    await programs_db.initialize_islands(num_islands, generator_temperature, model_name)

    # Run evolution in batches
    for batch in range(num_batches):
        await asyncio.gather(*[
            generate_recipes_for_island_batch(
                island, recipes_per_batch, deepinfra_api_key, template, generator_temperature, model_name
            )
            for island in programs_db.get_islands()
        ])

    return Island._programs

async def generate_recipes_for_island_batch(
    island,
    recipes_per_batch,
    deepinfra_api_key,
    template,
    generator_temperature,
    model_name
):
    """
    Generates and evaluates new recipes for a single island in one evolution batch.

    Args:
        island (Island): The island whose recipes will evolve.
        recipes_per_batch (int): Number of new recipe versions to generate.
        deepinfra_api_key (str): API key for the generation model.
        template (str): Contest instructions and formatting rules.
        generator_temperature (float): Creativity parameter for generation.
        model_name (str): Evaluation model identifier.
    """
    # Get top-ranked recipes from the island
    island_recipes = await get_formatted_recipes_for_island(island)
    top_candidates = sorted(island_recipes, key=lambda x: x['softmax_score'], reverse=True)[:2]

    # Initialize the evaluator
    evaluator = RecipeEvaluator(deepinfra_api_key)

    # Define evolution and scoring pipeline
    async def evolve_and_score():
        evolved_recipe = await evolve_function(top_candidates, deepinfra_api_key, template, generator_temperature)
        score = await evaluator.evaluate_recipe_with_llm(evolved_recipe, island.island_id, model_name)
        return evolved_recipe, score

    # Run multiple evolutions in parallel
    scored_recipes = await asyncio.gather(*[
        evolve_and_score() for _ in range(recipes_per_batch)
    ])

    # Register and evaluate evolved recipes in the island
    await asyncio.gather(*[
        island.register_and_evaluate_program(recipe, island.island_id, score)
        for recipe, score in scored_recipes
    ])
