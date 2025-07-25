class ProgramsDatabase:
    """A collection of generated programs, organized into multiple islands (clusters),
    with tools for initialization, evaluation, and ranking."""

    def __init__(self, template, deepinfra_api_key: str, functions_per_prompt: int = 5) -> None:
        # Prompt template for program generation
        self.template = template

        # API key for accessing the DeepInfra evaluation service
        self._deepinfra_api_key = deepinfra_api_key

        # List to store Island objects (each represents a subpopulation)
        self._islands = []

        # Number of programs generated per prompt within each island
        self._functions_per_prompt = functions_per_prompt

        # Flat list of all programs (not currently populated in this snippet)
        self.programs = []

    async def initialize_islands(self, num_islands, generator_temperature, model_name):
        """
        Concurrently initialize a set of islands with generated programs.

        - Each island is initialized with a unique seed and configuration.
        - Programs are generated using a language model with the given temperature and model name.
        - After generation, programs within each island are clustered.
        """
        tasks = []

        # Shared evaluator instance across islands
        evaluator = RecipeEvaluator(deepinfra_api_key=self._deepinfra_api_key)

        for i in range(num_islands):
            # Create and configure a new island
            island = Island(
                island_id=i,
                template=self.template,
                functions_per_prompt=self._functions_per_prompt,
                deepinfra_api_key=self._deepinfra_api_key,
                evaluator=evaluator
            )

            self._islands.append(island)

            # Generate a unique seed for reproducibility
            seed = str(uuid.uuid4())

            # Schedule asynchronous program initialization
            task = island.initialize_programs(seed, generator_temperature, model_name)
            tasks.append(task)

        # Run all island initializations concurrently
        await asyncio.gather(*tasks)

        # Perform clustering on the generated programs in each island
        for island in self._islands:
            island.cluster_programs()

    def rank_islands(self):
        """
        Rank the islands based on the highest scoring program within each island.

        - Uses a softmax distribution over the best scores to normalize and rank.
        - Returns islands sorted from most to least creative based on normalized scores.
        """
        island_scores = []
        for island in self._islands:
            best_program = island.get_best_program_from_clusters()
            island_scores.append(best_program['score'] if best_program else 0)

        # Normalize scores using softmax to emphasize relative differences
        softmax_island_scores = softmax(np.array(island_scores))

        # Sort islands by descending score
        ranked_islands = sorted(zip(self._islands, softmax_island_scores), key=lambda x: x[1], reverse=True)
        return ranked_islands

    def get_islands(self):
        """Return the list of all island objects (each containing its programs and clusters)."""
        return self._islands

    def get_best_programs(self) -> List[Dict[str, any]]:
        """Extract the best (highest scoring) program from each island."""
        return [island.get_best_program_from_clusters() for island in self._islands]
