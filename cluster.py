
class Cluster:
    """
    Handles clustering of programs based on their weighted score.

    Each cluster represents a group of similar programs (recipes) whose scores fall within
    a narrow range. The cluster tracks all associated programs and can return the best one
    based on softmax-normalized scores.
    """
    def __init__(self, score: float, program: Dict[str, any]) -> None:
        """
        Initialize a cluster with a single program and its score.

        Args:
            score (float): The weighted score of the initial program.
            program (Dict[str, any]): The initial program (e.g., recipe data).
        """

        self._score = score
        self._programs: List[Dict[str, any]] = [{'program': program, 'score': score}]
        self._lengths: List[int] = [len(str(program))]

    @property
    def score(self) -> float:
        """Returns the representative score of this cluster (used for matching)."""
        return self._score

    def add_program(self, program, score: float) -> None:
        """
        Adds a new program to the cluster and logs its inclusion.

        Args:
            program (Dict[str, any]): The program to add.
            score (float): The weighted score of the program.
        """
        self._programs.append({'program': program, 'score': score})  # Use _programs here
        logging.info(f"Added program with score: {score}")

    def register_program(self, program: Dict[str, any]) -> None:
        """
        Deprecated: Adds a raw program without score. Prefer `add_program()`.

        Args:
            program (Dict[str, any]): The program to add.
        """
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def get_best_program(self) -> Dict[str, any]:
        """
        Returns the best program in the cluster using softmax scoring.

        The program with the highest softmax-normalized score is chosen as the representative.

        Returns:
            Dict[str, any]: The best program's data and score.
        """
        # Step 1: Extract all raw scores
        scores = np.array([p['score'] for p in self._programs])
        # Step 2: Apply softmax to normalize scores
        softmax_scores = softmax(scores)
        # Step 3: Select the highest softmax score
        best_program_index = np.argmax(softmax_scores)
        return self._programs[best_program_index]
