async def main():
    """
    Orchestrates the full experimental pipeline for generating, selecting, and evaluating creative recipes.

    This function:
        - Iterates over generator temperatures and evaluator models.
        - Runs multiple optimization rounds per configuration.
        - Selects the best recipe from each run.
        - Formats recipes using an external formatter.
        - Evaluates formatted recipes using multiple models on TTCT dimensions.
        - Saves results, metrics, and final evaluation summaries to disk.
    """
    generator_temps = [0.5] # Temperature values to control creativity in generation
    model_names = ["bad"]  # Evaluator model identifiers (first = baseline, second = strong model)
    save_dir = '/content/drive/My Drive/Master_Thesis/experiment3/' # Output directory for results

    # Loop over all combinations of generator temp and evaluator model
    for generator in generator_temps:
        for model_name in model_names:
            print(f"\nProcessing generator={generator}, evaluator={model_name}")

            all_best_recipes = []

            for i in range(2):
                print(f"  -> Run {i+1}/30")

                # Repeat the evolutionary search multiple times for robustness
                results = await fun_search_optimization(
                    num_batches=5,
                    recipes_per_batch=5,
                    num_islands=7,
                    generator_temperature=generator,
                    model_name=model_name,
                    template=template,
                    deepinfra_api_key=deepinfra_api_key
                )

                # Run the recipe search and optimization process
                best_recipe = results.sort_values(by='weighted_score', ascending=False).iloc[0]
                best_recipe["generator"] = generator
                best_recipe["evaluator"] = model_name

                all_best_recipes.append(best_recipe)

            # Aggregate best recipes across all runs
            final_df = pd.DataFrame(all_best_recipes)

            # Format recipes using async formatter and external reference set
            async with aiohttp.ClientSession() as session:
                tasks = [async_formatter(session, row, pillsbury_recipes['formatted_recipe']) for _, row in final_df.iterrows()]
                final_df['better_format'] = await asyncio.gather(*tasks)
                print("final dataframe = ", final_df)

            # Save raw best recipe data
            setting_output_path = f= f"{save_dir}results{generator}_evaluator{model_name}_part2.csv"
            final_df.to_csv(setting_output_path, index=False)
            print(f"Saved best recipes for setting (generator={generator}, evaluator={model_name}) to {setting_output_path}")

            # Evaluate recipes' creativity using multiple models
            async def main_creative():
                result = await async_main(final_df)  # Use 'await' instead of 'asyncio.run()'
                print(result)
                return result

            result2, total_missing, out_of_range = await main_creative()
            print(result2)

            # Create a summary of evaluation errors (missing or out-of-range values)
            summary_df = pd.DataFrame({
                "metric": ["total_missing", "out_of_range"],
                "value": [total_missing, out_of_range]
            })

            # Combine evaluation results and summary metrics
            final_result = pd.concat([result2, summary_df], ignore_index=True)

            # Save creativity evaluation summary to CSV
            creativity = f"{save_dir}creativity_generator_{generator}_evaluator_{model_name}_part2.csv"
            final_result.to_csv(creativity, index=False)

            print(f"Saved results to {creativity}")



if __name__ == "__main__":
    asyncio.run(main())
