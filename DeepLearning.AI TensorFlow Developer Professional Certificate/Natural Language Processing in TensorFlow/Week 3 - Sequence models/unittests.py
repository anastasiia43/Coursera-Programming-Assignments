import numpy as np
from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
import tensorflow as tf


def parameter_count(model):
    total_params_solution, train_params_solution = 15_000_000, 35_000
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


def test_train_val_datasets(learner_func):
    def g():
        function_name = "train_val_datasets"

        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        t_size = 21000
        dummy_dataset = tf.data.Dataset.from_tensor_slices(
            (list(range(t_size)), list(range(t_size)))
        )

        train_dataset, validation_dataset = learner_func(dummy_dataset)

        t = test_case()
        if not isinstance(train_dataset, tf.data.Dataset):
            t.failed = True
            t.msg = "Incorrect output type for train_dataset"
            t.want = tf.data.Dataset
            t.got = type(train_dataset)
            return [t]

        t = test_case()
        if not isinstance(validation_dataset, tf.data.Dataset):
            t.failed = True
            t.msg = "Incorrect output type for validation_dataset"
            t.want = tf.data.Dataset
            t.got = type(validation_dataset)
            return [t]

        for text, label in train_dataset.take(1):
            training_text = text
            training_label = label

        for text, label in validation_dataset.take(1):
            validation_text = text
            validation_label = label

        t = test_case()
        if not isinstance(training_text, tf.Tensor):
            t.failed = True
            t.msg = "Incorrect output type for training texts"
            t.want = tf.Tensor
            t.got = type(training_text)
            return [t]

        t = test_case()
        if not isinstance(training_label, tf.Tensor):
            t.failed = True
            t.msg = "Incorrect output type for training labels"
            t.want = tf.Tensor
            t.got = type(training_label)
            return [t]

        t = test_case()
        if not isinstance(validation_text, tf.Tensor):
            t.failed = True
            t.msg = "Incorrect output type for validation texts"
            t.want = tf.Tensor
            t.got = type(validation_text)
            return [t]

        t = test_case()
        if not isinstance(validation_label, tf.Tensor):
            t.failed = True
            t.msg = "Incorrect output type for validation labels"
            t.want = tf.Tensor
            t.got = type(validation_label)
            return [t]

        t = test_case()
        if training_text.shape != 128:
            t.failed = True
            t.msg = "Incorrect dimension for tensor of training texts. Check that you set the correct batch size."
            t.want = 128
            t.got = training_text.shape.as_list()[0]
        cases.append(t)

        t = test_case()
        if validation_text.shape != 128:
            t.failed = True
            t.msg = "Incorrect dimension for tensor of validation texts. Check that you set the correct batch size."
            t.want = 128
            t.got = validation_text.shape.as_list()[0]
        cases.append(t)

        t = test_case()
        if training_label.shape != 128:
            t.failed = True
            t.msg = "Incorrect dimension for tensor of training labels. Check that you set the correct batch size."
            t.want = 128
            t.got = training_label.shape.as_list()[0]
        cases.append(t)

        t = test_case()
        if validation_label.shape != 128:
            t.failed = True
            t.msg = "Incorrect dimension for tensor of validation labels. Check that you set the correct batch size."
            t.want = 128
            t.got = validation_label.shape.as_list()[0]
        cases.append(t)

        splits = [0.9]
        num_batches = [128]

        for split, num_batch in zip(splits, num_batches):
            train, val = learner_func(dummy_dataset)
            train_size = 0
            val_size = 0
            for batch in train:
                train_size += len(batch)

            for batch in val:
                val_size += len(batch)

            t = test_case()
            if not np.isclose(train_size, int(round(t_size * split, 0)), 1):
                t.failed = True
                t.msg = f"Incorrect training size for a dataset of {t_size} elements, training_split = {split} and num_batches = {num_batch}"
                t.want = int(round(t_size * split, 0))
                t.got = train_size
            cases.append(t)

            t = test_case()
            if not np.isclose(val_size, t_size * (1 - split), 1):
                t.failed = True
                t.msg = f"Incorrect training size for a dataset of {t_size} elements and training_split = {split} and num_batches = {num_batch}"
                t.want = int(round(t_size * (1 - split), 0))
                t.got = val_size
            cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


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

        dummy_dataset = tf.data.Dataset.from_tensor_slices((["Test", "Not Test"]))

        vectorizer = learner_func(dummy_dataset)

        t = test_case()
        if not isinstance(vectorizer, tf.keras.layers.TextVectorization):
            t.failed = True
            t.msg = f"Got a wrong output type  for {function_name}"
            t.want = tf.keras.layers.TextVectorization
            t.got = type(vectorizer)
            return [t]

        t = test_case()
        if not len(vectorizer("this is a test")) == 32:
            t.failed = True
            t.msg = "Got a wrong sequence length for a test sentence. Make sure that MAX_LENGTH is set to 32 before submitting"
            t.want = 32
            t.got = len(vectorizer("this is a test"))
        cases.append(t)

        dummy_dataset = tf.data.Dataset.from_tensor_slices(
            (
                [
                    "my guitar is not a banjo you know?",
                    "cats and dogs and birds",
                    "hello my dudes",
                ]
            )
        )

        vectorizer = learner_func(dummy_dataset)
        vocab_size = vectorizer.vocabulary_size()

        t = test_case()
        if not vocab_size == 16:
            t.failed = True
            t.msg = "Incorrect vocabulary size for a corpus with 14 unique words ('' and '[UNK]' tokens should also be included)"
            t.want = 16
            t.got = vocab_size
        cases.append(t)

        vocab = vectorizer.get_vocabulary()
        expected_vocab = [
            "",
            "[UNK]",
            "my",
            "and",
            "you",
            "not",
            "know",
            "is",
            "hello",
            "guitar",
            "dudes",
            "dogs",
            "cats",
            "birds",
            "banjo",
            "a",
        ]
        sentences = [
            "my guitar is not a banjo you know?",
            "cats and dogs and birds",
            "hello my dudes",
        ]
        t = test_case()
        if not set(vocab) == set(expected_vocab):
            t.failed = True
            t.msg = f"Incorrect vocabulary for sentences:\n{sentences}"
            t.want = expected_vocab
            t.got = vocab
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

        EMBEDDING_DIM = 100

        VOCAB_SIZE = 145856
        model = learner_func(VOCAB_SIZE, np.zeros((VOCAB_SIZE, EMBEDDING_DIM)))

        last_2_layers = [layer for layer in model.layers[-2:]]

        final_layer = last_2_layers[-1]
        penultimate_layer = last_2_layers[0]

        t = test_case()
        if not isinstance(final_layer, tf.keras.layers.Dense):
            t.failed = True
            t.msg = "Incorrect type for last layer"
            t.want = tf.keras.layers.Dense
            t.got = type(final_layer)
        cases.append(t)

        t = test_case()
        if not isinstance(penultimate_layer, tf.keras.layers.Dense):
            t.failed = True
            t.msg = "Incorrect type for penultimate layer"
            t.want = tf.keras.layers.Dense
            t.got = type(penultimate_layer)
        cases.append(t)

        dropout_layers = [
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Dropout)
        ]

        t = test_case()
        if not dropout_layers:
            t.failed = True
            t.msg = "You must add at least one Dropout layer in your model"
            t.want = "At least one Dropout layer"
            t.got = "No Dropout layers"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_history(history):
    def g():
        cases = []

        t = test_case()
        if not isinstance(history, tf.keras.callbacks.History):
            t.failed = True
            t.msg = "history has incorrect type"
            t.want = tf.keras.callbacks.History
            t.got = type(history)
            return [t]

        if not history.history.get("loss"):
            t.failed = True
            t.msg = "history is missing 'loss' metric"
            t.want = "a metrics named 'loss'"
            t.got = None
            return [t]

        if not history.history.get("val_loss"):
            t.failed = True
            t.msg = "history is missing 'val_loss' metric"
            t.want = "a metrics named 'val_loss'"
            t.got = None
            return [t]

        val_loss = history.history["val_loss"]

        epochs = range(len(val_loss))

        val_acc_slope, _ = np.polyfit(np.array(epochs), np.array(val_loss), 1)

        t = test_case()
        if val_acc_slope >= 0.0005:
            t.failed = True
            t.msg = "maximum slope of validation loss exceeded"
            t.want = "a slope of 0.0005 at most"
            t.got = f"{val_acc_slope:.3f}%"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
