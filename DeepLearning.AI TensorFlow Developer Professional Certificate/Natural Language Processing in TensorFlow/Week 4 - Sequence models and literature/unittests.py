import numpy as np
from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
import tensorflow as tf


FILE_PATH = "./data/sonnets.txt"
NUM_BATCHES = 16
LSTM_UNITS = 128

# Read the data
with open(FILE_PATH) as f:
    data = f.read()

# Convert to lower case and save as a list
corpus = data.lower().split("\n")


def parameter_count(model):
    total_params_solution, train_params_solution = 2_000_000, 2_000_000
    total_params = model.count_params()
    num_trainable_params = sum(
        [w.shape.num_elements() for w in model.trainable_weights]
    )
    total_msg = f"\033[92mYour model has {total_params:,} total parameters and the reference is {total_params_solution:,}"
    train_msg = f"\033[92mYour model has {num_trainable_params:,} trainable parameters and the reference is {train_params_solution:,}"
    if total_params > total_params_solution:
        total_msg += f"\n\033[91mWarning! this exceeds the reference which is {total_params_solution:,}. If the kernel crashes while training, switch to a simpler architecture."
    else:
        total_msg += "\033[92m. You are good to go!"
    if num_trainable_params > train_params_solution:
        train_msg += f"\n\033[91mWarning! this exceeds the reference which is {train_params_solution:,}. If the kernel crashes while training, switch to a simpler architecture."
    else:
        train_msg += "\033[92m. You are good to go!"
    print(total_msg)
    print()
    print(train_msg)


def test_fit_vectorizer(learner_func):
    def g():
        function_name = "fit_vectorizer"
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        toy_corpus = [
            "I like coding with Tensorflow",
            "Making a tensorflow model makes me happy.",
        ]

        t = test_case()
        try:
            tokenizer = learner_func(toy_corpus)
        except Exception as e:
            t.failed = True
            t.msg = "Impossible to test your function due to exception being thrown"
            t.want = "No exceptions should be thrown"
            t.got = f"Following exception was thrown: {e}"
            return [t]

        t = test_case()
        if not isinstance(tokenizer, tf.keras.layers.TextVectorization):
            t.failed = True
            t.msg = "vectorizer is not a TextVectorization object (tf.keras.layers.TextVectorization)"
            t.want = "an instance of tf.keras.layers.TextVectorization"
            t.got = type(tokenizer)
            return [t]

        t = test_case()
        if tokenizer.get_config()["standardize"] != "lower_and_strip_punctuation":
            t.failed = True
            t.msg = "Incorrect standardize parameter"
            t.want = "lower_and_strip_punctuation"
            t.got = tokenizer.get_config()["standardize"]
        cases.append(t)

        # Removed because tf 2.16.1 removed this attribute
        #        t = test_case()
        #        if not tokenizer.is_adapted:
        #            t.failed = True
        #            t.msg = "TextVectorization is not adapted"
        #            t.want = "Call .adapt method in your code with the appropriate argument"
        #            t.got = "TextVectorization is not adapted"
        #            return [t]

        t = test_case()
        if len(tokenizer.get_vocabulary()) != 13:
            t.failed = True
            t.msg = f"Incorrect number of tokens in vocabulary when passing the following corpus:\n\t{test_corpus}\nDoublecheck if you passed the correct argument to standardize."
            t.want = "13 elements (including [UNK] token and empty token)"
            t.got = len(tokenizer.get_vocabulary())
        cases.append(t)

        test_sents = ["I like making a tensorflow model", "Tensorflow is awesome"]

        vect_sent_0 = tokenizer(test_sents[0])
        vect_sent_1 = tokenizer(test_sents[1])

        t = test_case()
        if len(vect_sent_0) == len(vect_sent_1):
            t.failed = True
            t.msg = "You've setup the TextVectorization layer to pad every sentence. Probably you set a value for pad_to_max_tokens or output_sequence_length"
            t.want = f"Output length for these sentences (when running the tokenizer separately on each sentence):\n\t{test_sents}\nhas the same output length but they should be distinct."
            t.got = (
                f"Same output for both sentences. Output length is: {len(vect_sent_0)}."
            )
        cases.append(t)

        t = test_case()
        if vect_sent_0.dtype != tf.int64:
            t.failed = True
            t.msg = "Incorrect dtype for output tensors."
            t.want = "Tensor dtype = tf.int64"
            t.got = vect_sent_0.dtype
        cases.append(t)
        return cases

    cases = g()
    print_feedback(cases)


def test_n_gram_seqs(learner_func):
    def g():
        function_name = "n_gram_seqs"

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        test_cases = [
            {
                "input": [
                    tf.convert_to_tensor(
                        [589, 457, 163, 583, 190, 641, 467], dtype=tf.int64
                    )
                ],
                "expected": [
                    tf.convert_to_tensor(x, dtype=tf.int64)
                    for x in [
                        [589, 457],
                        [589, 457, 163],
                        [589, 457, 163, 583],
                        [589, 457, 163, 583, 190],
                        [589, 457, 163, 583, 190, 641],
                        [589, 457, 163, 583, 190, 641, 467],
                    ]
                ],
            },
            {
                "input": [
                    tf.convert_to_tensor([783, 531], dtype=tf.int64),
                    tf.convert_to_tensor(
                        [893, 1674, 29834, 2456, 1539, 23467, 90843], dtype=tf.int64
                    ),
                ],
                "expected": [
                    tf.convert_to_tensor(x, dtype=tf.int64)
                    for x in [
                        [783, 531],
                        [893, 1674],
                        [893, 1674, 29834],
                        [893, 1674, 29834, 2456],
                        [893, 1674, 29834, 2456, 1539],
                        [893, 1674, 29834, 2456, 1539, 23467],
                        [893, 1674, 29834, 2456, 1539, 23467, 90843],
                    ]
                ],
            },
        ]

        cases = []

        dummy_tokenizer = lambda x: x

        for t_case in test_cases:
            t = test_case()
            try:
                learner_output = learner_func(t_case["input"], dummy_tokenizer)
            except Exception as e:
                t.failed = True
                t.msg = "Unittest aborted due to an execution error."
                t.want = f"Proper execution of function when passing the following tensors to split:\n\t{t_case['input']}"
                t.got = f"Thrown exception is: {e}"
                return [t]

            t = test_case()
            if not isinstance(learner_output, list):
                t.failed = True
                t.msg = "Incorrect output for {function_name}"
                t.want = "The output must be a list"
                t.got = type(learner_output)
                return [t]

            t = test_case()
            if len(learner_output) != len(t_case["expected"]):
                t.failed = True
                t.msg = (
                    f"Incorrect output size for input given by:\n\t{t_case['input']}"
                )
                t.want = f"Size must be {len(t_case['expected'])}"
                t.got = f"Size is: {len(learner_output)}"
            cases.append(t)

            learner_set = set([frozenset(list(x.numpy())) for x in learner_output])
            expected_set = set([frozenset(list(x.numpy())) for x in t_case["expected"]])

            t = test_case()
            if not learner_set == expected_set:
                t.failed = True
                t.msg = f"Incorrect output tensors for test case: {t_case['input']}"
                t.want = t_case["expected"]
                t.got = learner_output
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_pad_seqs(learner_func):
    def g():
        function_name = "pad_seqs"

        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        t = test_case()
        try:
            learner_output = learner_func([[1], [2], [3]], 3)
        except Exception as e:
            t.failed = True
            t.msg = f"Impossible to test function {function_name} due to an exception being thrown"
            t.want = "No exceptions when running the function"
            t.got = f"Exception was: {e}"
            return [t]

        if not isinstance(learner_output, np.ndarray):
            t.failed = True
            t.msg = "Incorrect output type"
            t.want = "A numpy array"
            t.got = type(learner_output)

        cases.append(t)

        test_seqs = [
            [[1], [1, 2], [1, 2, 3]],
            [[2, 3], [5, 6, 7, 8], [9, 0, 41, 6], [4, 7, 9]],
        ]
        maxlens = [6, 9]
        expected_outputs = [
            np.array(
                [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 2], [0, 0, 0, 1, 2, 3]],
                dtype=np.int32,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 2, 3],
                    [0, 0, 0, 0, 0, 5, 6, 7, 8],
                    [0, 0, 0, 0, 0, 9, 0, 41, 6],
                    [0, 0, 0, 0, 0, 0, 4, 7, 9],
                ],
                dtype=np.int32,
            ),
        ]

        for test_seq, maxlen, expected_output in zip(
            test_seqs, maxlens, expected_outputs
        ):
            t = test_case()
            try:
                learner_output = learner_func(test_seq, maxlen)
            except Exception as e:
                t.failed = True
                t.msg = f"Aborting test due to an exception being thrown with inputs: input_sequences =\n{test_seq} and maxlen =\n{maxlen}"
                t.want = "No exception"
                t.got = f"Exception thrown: {e}"
                return [t]

            if not np.allclose(learner_output, expected_output):
                t.failed = True
                t.msg = f"Incorrect output if input_sequences =\n{test_seq} and maxlen =\n{maxlen}"
                t.want = expected_output
                t.got = learner_output
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_features_and_labels_dataset(learner_func):
    def g():
        function_name = "features_and_labels_dataset"

        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        t = test_case()
        try:
            np.random.seed(44)
            input_sequences = np.random.randint(0, 1000, size=(30, 45))
            maxlen = 1001
            learner_output = learner_func(input_sequences, maxlen)
        except Exception as e:
            t.failed = True
            t.msg = f"Impossible to test function {function_name} due to an exception being thrown"
            t.want = "No exceptions when running the function"
            t.got = f"Exception was: {e}"
            return [t]
        if not isinstance(learner_output, tf.data.Dataset):
            t.failed = True
            t.msg = "Output is not a tensorflow dataset (tf.data.Dataset)"
            t.want = "An instance of a tensorflow dataset"
            t.got = type(learner_output)
        cases.append(t)

        t = test_case()
        if "BatchDataset" not in learner_output.__class__.__name__:
            t.failed = True
            t.msg = "Train dataset is not batched."
            t.want = "Batched Dataset"
            t.got = f"{learner_output.__class__.__name__}"
            cases.append(t)
            return cases
        cases.append(t)

        test_input_sequence = np.array(
            [
                [669, 622, 392, 194, 504, 816],
                [678, 87, 472, 757, 740, 159],
                [21, 241, 397, 457, 924, 910],
                [842, 841, 998, 277, 740, 881],
                [561, 661, 762, 86, 673, 666],
                [292, 47, 655, 812, 618, 873],
                [283, 0, 377, 706, 319, 117],
                [962, 116, 213, 630, 562, 82],
                [408, 943, 718, 538, 236, 306],
                [496, 87, 89, 360, 573, 292],
            ]
        )
        test_total_words = 911

        expected_features = np.array(
            [
                [669, 622, 392, 194, 504],
                [678, 87, 472, 757, 740],
                [21, 241, 397, 457, 924],
                [842, 841, 998, 277, 740],
                [561, 661, 762, 86, 673],
                [292, 47, 655, 812, 618],
                [283, 0, 377, 706, 319],
                [962, 116, 213, 630, 562],
                [408, 943, 718, 538, 236],
                [496, 87, 89, 360, 573],
            ]
        )

        expected_labels_shape = (10, 911)

        learner_output = learner_func(test_input_sequence, test_total_words)

        t = test_case()
        if len(learner_output) != 1:
            t.failed = True
            t.msg = f"Incorrect number of batches for input = {test_input_sequence}, total_words = {test_total_words}"
            t.want = 1
            t.got = len(learner_output)
        cases.append(t)

        t = test_case()
        for feature, label in learner_output:
            if feature.shape != (10, 5):
                t.failed = True
                t.msg = f"Incorrect batch feature size for input = {test_input_sequence}, total_words = {test_total_words}. Aborting test."
                t.want = (5, 5)
                t.got = feature.shape
                cases.append(t)
                return cases
            cases.append(t)
            if not np.allclose(feature.numpy(), expected_features):
                t.failed = True
                t.msg = f"Incorrect feature output in batch 1 for input = {test_input_sequence}, total_words = {test_total_words}"
                t.want = expected_features
                t.got = feature.numpy()

            t = test_case()
            if label.shape != expected_labels_shape:
                t.failed = True
                t.msg = f"Incorrect label shape in batch 1 for input = {test_input_sequence}, total_words = {test_total_words}"
                t.want = expected_labels_shape
                t.got = label.shape
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_create_model(learner_func):
    def g():
        function_name = "create_model"

        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        total_words = 3189
        max_sequence_len = 11

        t = test_case()
        try:
            model = learner_func(total_words, max_sequence_len)
        except Exception as e:
            t.failed = True
            t.msg = f"Impossible to test function {function_name} due to an exception being thrown"
            t.want = "No exceptions when running the function"
            t.got = f"Exception was: {e}"
            return [t]

        t = test_case()
        x = np.zeros((16, 10))
        y = np.zeros((16, 3189))
        try:
            model.evaluate(x, y, verbose=False)
        except Exception as e:
            t.failed = True
            t.msg = "Your model is not compatible with the dataset you defined earlier. Check that the loss function, last layer and label_mode are compatible with one another"
            t.want = "Your model should be able to predict/evaluate with input shape (None,10) and label shape (None, 3189) where None is the number of elements in the batch"
            t.got = f"The following exception was thrown:\n{e}"
            return [t]
        cases.append(t)

        first_layer = model.layers[0]

        t = test_case()
        if not isinstance(first_layer, tf.keras.layers.Embedding):
            t.failed = True
            t.msg = "First layer must be an Embedding layer as provided in the code"
            t.want = "First layer as Embedding layer"
            t.got = f"First layer is: {first_layer}"
            return [t]
        cases.append(t)

        t = test_case()
        if first_layer.input.shape[1] != 10:
            t.failed = True
            t.msg = "Incorrect input shape."
            t.want = "10"
            t.got = first_layer.input.shape[1]

        t = test_case()
        # Sum returns 0 if there is no LSTM nor Bidirectional defined and positive integer otherwise
        if not sum(
            [
                isinstance(layer, tf.keras.layers.LSTM)
                or isinstance(layer, tf.keras.layers.Bidirectional)
                for layer in model.layers
            ]
        ):
            t.failed = True
            t.msg = "No LSTM or Bidirectional layer defined in your model"
            t.want = "At least one layer must be an LSTM or Bidirectional layer"
            t.got = "No LSTM nor Bidirectional has been found"
        cases.append(t)

        t = test_case()
        # In case there is a Bidirectional, checks if the layer within is an LSTM
        if sum(
            [isinstance(layer, tf.keras.layers.Bidirectional) for layer in model.layers]
        ) and not sum(
            [
                isinstance(bidirectional_layer.forward_layer, tf.keras.layers.LSTM)
                for bidirectional_layer in model.layers
                if isinstance(bidirectional_layer, tf.keras.layers.Bidirectional)
            ]
        ):
            t.failed = True
            t.msg = "Layer within at least Bidirectional layer must be an LSTM"
            t.want = "At least one bidirectional layer must be made of an LSTM"
            t.got = "No bidirectional layer is made of an LSTM"
        cases.append(t)

        last_layer = model.layers[-1]

        t = test_case()
        if not isinstance(last_layer, tf.keras.layers.Dense):
            t.failed = True
            t.msg = "Last layer must be a dense layer"
            t.want = "Last layer must be an instance of tf.keras.layers.Dense"
            t.got = f"Last layer is of type: {type(last_layer)}"
            cases.append(t)
            return cases
        cases.append(t)

        t = test_case()
        if model.layers[-1].units != 3189:
            t.failed = True
            t.msg = (
                "Number of units in last Dense layer must be the same as the vocabulary"
            )
            t.want = "Same number of units as the total word count"
            t.got = f"Last Dense layer has {model.layers[-1].units} units"
        cases.append(t)

        t = test_case()
        if not isinstance(
            model.layers[-1].activation, type(tf.keras.activations.softmax)
        ):
            t.failed = True
            t.msg = "Last Dense layer activation must be a softmax"
            t.want = "Activation of last layer must be softmax"
            t.got = model.layers[-1].activation
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
