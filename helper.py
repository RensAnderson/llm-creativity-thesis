def softmax(scores):
    """
    Apply softmax normalization to a list or array of scores.

    This is used to convert raw scores into a probability-like distribution,
    emphasizing relative differences while preserving ordering.

    Args:
        scores (np.ndarray): Array of numeric scores.

    Returns:
        np.ndarray: Softmax-normalized scores.
    """
    exp_scores = np.exp(scores - np.max(scores))  # Stabilize for numerical safety
    return exp_scores / exp_scores.sum()


async def get_formatted_recipes_for_island(island):
    """
    Retrieve and format recipes for a given island with softmax-based ranking.

    Args:
        island (Island): An Island object containing programs/recipes.

    Returns:
        List[Dict[str, any]]: List of formatted recipe dictionaries.
    """
    # Rank the island's programs by softmax-normalized scores
    ranked_programs = island.rank_programs()

    formatted_recipes = []

    for _, recipe in ranked_programs.iterrows():
        formatted_recipes.append({
            "recipe_idea": recipe["recipe_idea"],
            "essay": recipe["essay"],
            "recipe_name": recipe["recipe_name"],
            "ingredients": recipe["ingredients"],
            "instructions": recipe["instructions"],
            "softmax_score": recipe.get("softmax_score", 0)
        })

    return formatted_recipes


async def async_formatter(session, row, pillsbury_examples):
    """
    Format a raw recipe to match Pillsbury's style using LLM inference.

    Args:
        session (aiohttp.ClientSession): Active HTTP session for API calls.
        row (pd.Series): Row from a DataFrame containing recipe fields.
        pillsbury_examples (List[str]): Two example formatted recipes for context.

    Returns:
        str: The formatted recipe text or an error message.
    """
    # Extract raw components from the row
    title = row['recipe_name']
    ingredients = row['ingredients']
    instructions = row['instructions']

    # Construct the prompt using examples and the raw recipe
    prompt = f"""
    You should return a recipe with the exact format as {pillsbury_examples[0]} and {pillsbury_examples[1]}.
    The title of the recipe is {title}.
    The ingredients of the recipe are {ingredients}.
    The instructions of the recipe are {instructions}.
    Only return the recipe and nothing else!
    """

    headers = {
        "Authorization": f"Bearer {deepinfra_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
        "max_tokens": 512
    }

    url = "https://api.deepinfra.com/v1/openai/chat/completions"

    try:
        # Make asynchronous API call to DeepInfra
        async with session.post(url, json=data, headers=headers) as response:
            response.raise_for_status()
            response_json = await response.json()

            # Return only the generated recipe content
            return response_json["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"Error formatting recipe: {e}")
        return "Error or no response from API."
