
class PartialParse(object):
    def __init__(self, sentence):
        """Initialize a partial parse.

        Args:
            sentence: The sentence to be parsed as a list of words.
        """
        # The sentence being parsed is kept for bookkeeping purposes
        self.sentence = sentence

        # Initialize the stack with ROOT
        self.stack = ["ROOT"]
        # Initialize buffer with all words in the sentence
        self.buffer = list(sentence.copy())
        # Initialize dependencies as empty list
        self.dependencies = []

    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        Args:
            transition: A string that equals "S", "LA", or "RA" representing the shift,
                        left-arc, and right-arc transitions.
        """
        if transition == "S":  # Shift
            # Remove the first word from buffer and push it onto the stack
            if self.buffer:
                self.stack.append(self.buffer.pop(0))

        elif transition == "LA":  # Left Arc
            # Mark second item on stack as dependent of first item and remove second item
            if len(self.stack) >= 2:
                dependent = self.stack[-2]
                head = self.stack[-1]
                self.dependencies.append((head, dependent))
                self.stack.pop(-2)

        elif transition == "RA":  # Right Arc
            # Mark first item on stack as dependent of second item and remove first item
            if len(self.stack) >= 2:
                dependent = self.stack[-1]
                head = self.stack[-2]
                self.dependencies.append((head, dependent))
                self.stack.pop(-1)


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    Args:
        sentences: A list of sentences to be parsed
        model: The model that makes parsing decisions
        batch_size: The number of PartialParses to include in each minibatch

    Returns:
        dependencies: A list where each element is the dependencies list for a parsed sentence.
    """
    # Initialize partial parses
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses.copy()

    # Continue until all parses are complete
    while unfinished_parses:
        # Get the first batch_size parses
        minibatch = unfinished_parses[:batch_size]

        # Use the model to predict the next transition for each parse in the minibatch
        transitions = model.predict(minibatch)

        # Apply the predicted transitions
        for i, transition in enumerate(transitions):
            minibatch[i].parse_step(transition)

        # Remove completed parses
        unfinished_parses = [p for p in unfinished_parses
                             if len(p.buffer) > 0 or len(p.stack) > 1]

    # Return the dependencies for each completed parse
    return [p.dependencies for p in partial_parses]