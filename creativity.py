def generate_prompt(recipe):
    """
    Constructs a prompt to evaluate a recipe's creativity based on the Torrance Tests of Creative Thinking (TTCT).

    Args:
        recipe (str): A formatted recipe description string.

    Returns:
        str: A formatted prompt string structured for model evaluation of creativity.
    """
    return f"""You are an expert in evaluating recipe creativity based on the Torrance Tests of Creative Thinking (TTCT). Your task is to assess the following recipe according to the four dimensions of creativity: Fluency, Flexibility, Elaboration, and Originality.

    Recipe: '{recipe}'

    Please evaluate this recipe based on the following dimensions:

    1. **Fluency** – Does the recipe contain multiple distinct creative elements, such as innovative ingredient combinations or unique preparation techniques? (Score: 1 = low, 5 = high)
    2. **Flexibility** – Does the recipe showcase versatility in ingredient use, cooking methods, or cultural fusion? (Score: 1 = low, 5 = high)
    3. **Elaboration** – How well does the recipe provide depth, explanation, and clarity in its preparation steps and ingredient choices? (Score: 1 = low, 5 = high)
    4. **Originality** – How unique is this recipe compared to traditional versions? Does it introduce new concepts, techniques, or ingredient uses? (Score: 1 = low, 5 = high)

    For each dimension, please:
    - First, provide a **detailed qualitative assessment** of the recipe’s creativity.
    - Immediately **assign a score** from 1 to 5 (1 being low creativity, 5 being high creativity) for each dimension.
    - The **score must be included immediately after the qualitative assessment**.

    Your response **must follow this exact JSON format below** with no additional explanations, comments, or text. Do not include any other tags like `</instructions>` or `</output_format>`. The response should be **only** the JSON object.

    <output_format>
    {{
        "fluency_quality_assess": "Your qualitative assessment for Fluency",
        "fluency": "Score between 1-5",
        "flexibility_quality_assess": "Your qualitative assessment for Flexibility",
        "flexibility": "Score between 1-5",
        "elaboration_quality_assess": "Your qualitative assessment for Elaboration",
        "elaboration": "Score between 1-5",
        "originality_quality_assess": "Your qualitative assessment for Originality",
        "originality": "Score between 1-5"
    }}
    </output_format>
    """



async def call_model_api(model, prompt, max_retries=3):
    """
    Calls the OpenAI ChatCompletion API asynchronously, with retry logic and exponential backoff.

    Args:
        model (str): Name of the OpenAI model (e.g., "gpt-4o-mini").
        prompt (str): Prompt to send for evaluation.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.

    Returns:
        str or None: The model's response text if successful, otherwise None.
    """
    for attempt in range(max_retries):
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.25
            )
            response_text = response['choices'][0]['message']['content'].strip()
            if response_text:
                return response_text
            else:
                logger.warning(f"Empty response from {model}, retrying... (Attempt {attempt+1})")

        except Exception as e:
            logger.error(f"Error calling {model}: {e}. Retrying... (Attempt {attempt+1})")

        await asyncio.sleep(2 ** attempt)  # Exponential backoff

    logger.error(f"Final failure: No valid response from {model}. Returning None.")
    return None



async def call_deepinfra_async(model, prompt):
    """
    Sends an asynchronous request to the DeepInfra API for a specified model.

    Args:
        model (str): Short model alias (e.g., "llama", "phi", "meta").
        prompt (str): The input prompt for generation.

    Returns:
        str: The generated text result from DeepInfra, or an empty string if failed.
    """
    model_mapping = {
        "llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "phi": "microsoft/Phi-4-multimodal-instruct",
        "meta": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    }
    deepinfra_model = model_mapping.get(model, model)

    headers = {
        "Authorization": f"Bearer {deepinfra_api_key}",
        "Content-Type": "application/json"
    }
    url = f"https://api.deepinfra.com/v1/inference/{deepinfra_model}"
    data = {
        "input": prompt,
        "temperature": 0.25,
        "max_tokens": 90
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    response_json = await response.json()
                    if "results" in response_json and isinstance(response_json["results"], list):
                        answer = response_json["results"][0].get("generated_text", "").strip()
                        return answer
                logger.error(f"API request failed with status {response.status}")
    except aiohttp.ClientError as e:
        logger.error(f"Network error calling DeepInfra API: {e}")
    return ""

async def retry_api_call(func, *args, retries=3, delay=2):
    """
    Executes an async function with exponential backoff retry logic.

    Args:
        func (Callable): The asynchronous function to call.
        *args: Arguments to pass to the function.
        retries (int, optional): Number of retry attempts. Defaults to 3.
        delay (int, optional): Initial delay in seconds for backoff. Defaults to 2.

    Returns:
        Any: Result of the async function, or None if all retries fail.
    """
    for attempt in range(retries):
        try:
            return await func(*args)
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(f"API call failed after {retries} attempts: {e}")
                return ""

def extract_scores_from_response(response_text, recipe_name, model_name):
    """
    Parses model response text to extract TTCT creativity scores using JSON and regex fallbacks.

    Args:
        response_text (str): Raw response from the language model.
        recipe_name (str): Identifier of the recipe.
        model_name (str): Name of the model that generated the response.

    Returns:
        list: A list containing [recipe_name, model_name, fluency, flexibility, elaboration, originality, average_score].
    """
    score_keys = ["fluency", "flexibility", "elaboration", "originality"]
    scores = {}
    missing_count = 0  # Track missing values per response

    # Try extracting JSON-formatted scores
    json_match = re.search(r"\{[\s\S]*?\}", response_text)
    if json_match:
        json_text = json_match.group(0).strip()
        try:
            response_json = json.loads(json_text)

            for key in score_keys:
                if key in response_json and isinstance(response_json[key], (int, float)) and 1 <= response_json[key] <= 5:
                    scores[key] = float(response_json[key])

        except json.JSONDecodeError:
            pass  # Ignore JSON errors

    # Try extracting scores using regex if some are missing
    if len(scores) < 4:
        for key in score_keys:
            pattern = rf'"?\b{key}\b"?\s*[:=]\s*"?([1-5](?:\.\d+)?)"?'
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match and key not in scores:
                try:
                    scores[key] = float(match.group(1))
                except ValueError:
                    pass

    # Final fallback: simple pattern matching if any scores are still missing
    if len(scores) < 4:
        for key in score_keys:
            pattern = rf"{key}.*?\b([1-5])\b"
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match and key not in scores:
                try:
                    scores[key] = float(match.group(1))
                except ValueError:
                    pass


    # Compute average only if all four scores are present
    average_score = sum(scores[key] for key in score_keys) / len(score_keys) if len(scores) == 4 else None

    return [recipe_name, model_name] + [scores.get(key, None) for key in score_keys] + [average_score]

async def process_recipes_async(df_recipes, models):
    """
    Iterates over all recipes and models, evaluates them asynchronously, and collects TTCT scores.

    Args:
        df_recipes (pd.DataFrame): DataFrame with recipe entries to process.
        models (list[str]): List of model identifiers (e.g., ["gpt-4o-mini", "llama"]).

    Returns:
        list[list]: List of scored rows per recipe-model pair, each with 7 values.
    """
    results = []
    tasks = []
    counter = 0

    for _, row in df_recipes.iterrows():
        formatted_prompt = generate_prompt(row['better_format'])
        for model in models:
            if model in ["gpt-4o-mini"]:
                task = retry_api_call(call_model_api, model, formatted_prompt)
            elif model in ["llama", "phi", "meta"]:
                task = retry_api_call(call_deepinfra_async, model, formatted_prompt)
            else:
                logger.warning(f"Unknown model: {model}. Skipping.")
                continue
            tasks.append((task, row, model))

    responses = await asyncio.gather(*(t[0] for t in tasks))

    for (task, row, model), response in zip(tasks, responses):
        scores = extract_scores_from_response(response, row['recipe_name'], model)
        if scores:
            results.append(scores)  # Append the first 5 scores

    return results

async def async_main(df_recipes):
    """
    Main asynchronous orchestration function to process recipes and return final results.

    Args:
        df_recipes (pd.DataFrame): DataFrame containing input recipes with formatting applied.

    Returns:
        tuple:
            pd.DataFrame: Final results including per-model scores and averages.
            int: Total number of missing score values.
            int: Total number of out-of-range score values.
    """
    models = ['llama', 'phi', 'meta', 'gpt-4o-mini']

    raw_results = await process_recipes_async(df_recipes, models)
    df_results = pd.DataFrame(raw_results, columns=['recipe', 'model', 'fluency', 'flexibility', 'originality', 'elaboration', 'average_score'])

    # Convert score columns to numeric
    score_columns = ['fluency', 'flexibility', 'originality', 'elaboration', 'average_score']
    df_results[score_columns] = df_results[score_columns].apply(pd.to_numeric, errors='coerce')

    # Compute averages per recipe
    df_averages = df_results.groupby('recipe')[score_columns].mean().reset_index()
    df_averages['model'] = 'Average'

    # Concatenate original results with averages
    df_final_sorted = pd.concat([df_results, df_averages], ignore_index=True)

    # Sort by model rank
    model_order = {'gpt-4o-mini': 0, 'llama': 1, 'phi': 2, 'meta': 3, 'Average': 4}
    df_final_sorted['model_rank'] = df_final_sorted['model'].map(model_order)
    df_final_sorted = df_final_sorted.sort_values(by=['recipe', 'model_rank']).drop(columns=['model_rank'])
    df_final_sorted = df_final_sorted.drop_duplicates(subset=['recipe', 'model'])
    df_final_sorted[score_columns] = df_final_sorted[score_columns].apply(pd.to_numeric, errors='coerce')

    # Count missing values
    total_missing = df_final_sorted.isna().sum().sum()

    # Count out-of-range values (less than 1 or greater than 5)
    out_of_range = ((df_final_sorted[score_columns] < 1) | (df_final_sorted[score_columns] > 5)).sum().sum()

    return df_final_sorted, total_missing, out_of_range
