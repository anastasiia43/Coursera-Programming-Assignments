import numpy as np
from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
import tensorflow as tf
from dlai_grader.io import suppress_stdout_stderr
import re

VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
TRAINING_SPLIT = 0.8


def parameter_count(model):
    total_params_solution, train_params_solution = 20_000, 20_000
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
        
        data = np.loadtxt('data/bbc-text.csv', delimiter=',', skiprows=1, dtype='str', comments=None)
        t = test_case()
        try:
            train_dataset, validation_labels = learner_func(data)
        except Exception as e:
            t.failed = True
            t.msg = f"There was an error evaluating the {function_name} function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not isinstance(train_dataset, tf.data.Dataset):
            t.failed = True
            t.msg = "Got a wrong output type for train_dataset."
            t.want = tf.data.Dataset
            t.got = type(train_dataset)
            return [t]
        
        t = test_case()
        if not isinstance(validation_labels, tf.data.Dataset):
            t.failed = True
            t.msg = "Got a wrong output type for validation_labels"
            t.want = tf.data.Dataset
            t.got = type(validation_labels)
            return [t]
        
        for text, label in train_dataset.take(1):
            training_text = text
            training_label = label

        for text, label in validation_labels.take(1):
            val_text = text
            val_label = label

        # Test types of the output object
        t = test_case()
        if not training_text.dtype == 'string':
            t.failed = True
            t.msg = "Got a wrong data type for texts in the train dataset"
            t.want = 'string'
            t.got = training_text.dtype
        cases.append(t)

        t = test_case()
        if not training_label.dtype == 'string':
            t.failed = True
            t.msg = "Got a wrong data type for labels in the train dataset"
            t.want = 'string'
            t.got = training_label.dtype
        cases.append(t)

        t = test_case()
        if not val_text.dtype == 'string':
            t.failed = True
            t.msg = "Got a wrong data type for texts in the validation dataset"
            t.want = 'string'
            t.got = val_text.dtype
        cases.append(t)

        t = test_case()
        if not val_label.dtype == 'string':
            t.failed = True
            t.msg = "Got a wrong data type for labels in the validation dataset"
            t.want = 'string'
            t.got = val_label.dtype
        cases.append(t)


        # Test 0.8 split with full bbc dataset
        t = test_case()
        if not train_dataset.cardinality() == 1780:
            t.failed = True
            t.msg = "Got wrong number of data points in the train dataset.  Make sure that TRAINING_SPLIT is set to 0.8 before submitting"
            t.want = 1780
            t.got = train_dataset.cardinality()
        cases.append(t)

        t = test_case()
        if not validation_labels.cardinality() == 445:
            t.failed = True
            t.msg = "Got wrong number of data points in the validation dataset. Make sure that TRAINING_SPLIT is set to 0.8 before submitting"
            t.want = 445
            t.got = validation_labels.cardinality()
        cases.append(t)

        return cases
    
    cases = g()
    print_feedback(cases)


def test_fit_vectorizer(learner_func, standardize_func):
    def g():
        function_name = "fit_vectorizer"

        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "fit_vectorizer has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        
        
        t = test_case()
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
        data = np.loadtxt('data/bbc-text.csv', delimiter=',', skiprows=1, dtype='str', comments=None)
        data = tf.data.Dataset.from_tensor_slices(data[:,1])
        
        try:
            vectorizer = learner_func(data, standardize_func)
        except Exception as e:
            t.failed = True
            t.msg = f"There was an error evaluating the {function_name} function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not isinstance(vectorizer, tf.keras.layers.TextVectorization):
            t.failed = True
            t.msg = f"Got a wrong output type  for {function_name}"
            t.want = tf.keras.layers.TextVectorization
            t.got = type(vectorizer)
            return [t]
        
        # test vocab size
        t = test_case()
        if not vectorizer.vocabulary_size() == 1000:
            t.failed = True
            t.msg = "Got a wrong number of elements in the vocabulary. Make sure that VOCAB_SIZE is set to 1000 before submitting"
            t.want = 1000
            t.got = vectorizer.vocabulary_size()
        cases.append(t)
        
        vocabulary = vectorizer.get_vocabulary()
        # test all stopwords are removed
        t = test_case()
        if any(np.in1d(vocabulary,stopwords)):
            t.failed = True
            t.msg = "Found stopwords in the vocabulary"
            t.want = "No stopwords in the vocabulary"
            found_stopwords = [vocabulary[k] for k in np.where(np.in1d(vocabulary,stopwords))[0]]
            t.got = f"Found the stopwords {found_stopwords}"
        cases.append(t)

        # test all punctuation is stripped
        t = test_case()
        vocabulary_noOOV = vocabulary.copy()
        vocabulary_noOOV.remove('[UNK]') # remove OOV token, because it has []
        # Get punctution from each word in vocabulary 
        all_punct = [re.findall(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", voc ) for voc in vocabulary_noOOV]
        # Flatten list and remove duplicates
        unique_punct = np.unique(sum(all_punct,[]))
        if not len(unique_punct)==0:#sum(any_punct)==0:
            t.failed = True
            t.msg = "Found punctuation in the vocabulary"
            t.want = "No punctuation in the vocabulary"
            #found_punct = [vocabulary_noOOV[k] for k in np.where(any_punct)[0]]
            t.got = f"Found the following punctuation: {np.unique(unique_punct)}"
        cases.append(t)

        t = test_case()
        if not len(vectorizer("this is a test")) == 120:
            t.failed = True
            t.msg = "Got a wrong sequence length for a test sentence. Make sure that MAX_LENGTH is set to 120 before submitting"
            t.want = 120
            t.got = len(vectorizer("this is a test"))
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_fit_label_encoder(learner_func):
    def g():
        function_name = "fit_label_encoder"

        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(function_name)
            return [t]
        
        t = test_case()
        labels_arr = np.loadtxt('data/bbc-text.csv', delimiter=',', skiprows=1, dtype='str', usecols=0, comments=None)
        train_labels = tf.data.Dataset.from_tensor_slices(labels_arr[:-1])
        validation_labels = tf.data.Dataset.from_tensor_slices(labels_arr[-1:])

        try:
            lab_encoder = learner_func(train_labels, validation_labels)
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating the fit_label_encoder function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not isinstance(lab_encoder, tf.keras.layers.StringLookup):
            t.failed = True
            t.msg = "Got a wrong output type for fit_label_encoder"
            t.want = tf.keras.layers.StringLookup
            t.got = type(lab_encoder)
            return [t]
        
        t = test_case()
        if not all(np.in1d(lab_encoder.get_vocabulary(), ['sport', 'business', 'politics', 'tech', 'entertainment'])): 
            t.failed = True
            if '[UNK]' in lab_encoder.get_vocabulary(): 
                t.failed = True
                t.msg = 'Found an OOV token in the labels vocabulary'
                t.want = "['sport', 'business', 'politics', 'tech', 'entertainment']" 
                t.got = lab_encoder.get_vocabulary()
            else: 
                t.msg = "Got the wrong vocabulary to encode labels"
                t.want = "['sport', 'business', 'politics', 'tech', 'entertainment']"
                t.got = lab_encoder.get_vocabulary()
        cases.append(t)

        t = test_case()
        if lab_encoder.vocabulary_size()<5:
            t.failed = True
            t.msg = 'You have missing labels'
            t.want = "5 labels: ['sport', 'business', 'politics', 'tech', 'entertainment']"
            t.got = f"{lab_encoder.vocabulary_size()} lables: {lab_encoder.vocabulary_size()} labels: {lab_encoder.get_vocabulary()}"
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

def test_preprocess_dataset(learner_func, text_vectorizer, lab_encoder):
    def g():
        function_name = "preprocess_dataset"

        cases = []    

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        data = np.loadtxt('data/bbc-text.csv', delimiter=',', skiprows=1, dtype='str', comments=None)
        dataset = tf.data.Dataset.from_tensor_slices((data[:,1], data[:,0]))
        
        try:
            proc_dataset = learner_func(dataset, text_vectorizer, lab_encoder)
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating the preprocess_dataset function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not isinstance(proc_dataset, tf.data.Dataset):
            t.failed = True
            t.msg = "Got a wrong output type for preprocess_dataset"
            t.want = tf.data.Dataset
            t.got = type(proc_dataset)
            return [t]
        
        batch = next(proc_dataset.as_numpy_iterator())
        
        t = test_case()
        batch_size = len(batch[0])
        if not batch_size == 32:
            t.failed = True
            t.msg = "Got wrong batch size"
            t.want = 32
            t.got = batch_size
        cases.append(t)

        t = test_case()
        if not isinstance(batch[0], np.ndarray):
            t.failed = True
            t.msg = "Got wrong type por the preprocessed texts"
            t.want = np.ndarray
            t.got = type(batch[0])
        cases.append(t)

        t = test_case()
        if not isinstance(batch[1], np.ndarray):
            t.failed = True
            t.msg = "Got wrong type for the preprocessed labels"
            t.want = np.ndarray
            t.got = type(batch[1])
        cases.append(t)

        t = test_case()
        if not batch[1].dtype == 'int64':
            t.failed = True
            t.msg = "Got wrong data type for the preprocessed texts"
            t.want = 'int64'
            t.got = batch[0].dtype
        cases.append(t)

        t = test_case()
        if not batch[1].dtype == 'int64':
            t.failed = True
            t.msg = "Got wrong data type for the preprocessed labels"
            t.want = 'int64'
            t.got = batch[1].dtype
        cases.append(t)

        t = test_case()
        if not batch[0].shape == (32, 120):
            t.failed = True
            t.msg = "Got wrong shape for the preprocessed texts. Make sure that MAX_LENGTH is set to 120 before submitting"
            t.want = (32, 120)
            t.got = batch[0].shape
        cases.append(t)

        t = test_case()
        if not batch[1].shape == (32,):
            t.failed = True
            t.msg = "Got wrong shape for the embedded labels"
            t.want = (32, )
            t.got = batch[1].shape
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
        
        fake_dataset = tf.data.Dataset.from_tensor_slices(([[2]*MAX_LENGTH],
                                                           [[1]])).batch(1)

        try:
            with suppress_stdout_stderr():
                model = learner_func()
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating the create_model function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not isinstance(model, tf.keras.models.Sequential):
            t.failed = True
            t.msg = "Got a wrong output type for create_model"
            t.want = tf.keras.models.Sequential
            t.got = type(model)
            return [t]

        t = test_case()
        try:
            with suppress_stdout_stderr():
                hist = model.fit(fake_dataset, epochs=1, validation_data= fake_dataset)
        except Exception as e:
            t.failed = True
            t.msg = "There was an error trying to fit the model"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        try:
            model_input = model.inputs[0]
        except Exception as e:
            t.failed = True
            t.msg = "your model is missing the Input"
            t.want = "a model with a defined Input"
            t.got = str(e)
            return [t]
        
        t = test_case()
        if not isinstance(model_input, tf.keras.KerasTensor):
            t.failed = True
            t.msg = "the input of your model has incorrect type"
            t.want = "a tf.keras.KerasTensor defined via tf.keras.Input"
            t.got = model_input
            return [t]
        
        input_shape = model.input_shape
        t = test_case()
        if input_shape != (None, MAX_LENGTH):
            t.failed = True
            t.msg = "model has incorrect input_shape"
            t.want = (None, 120)
            t.got = input_shape
            return [t]
        
        t = test_case()
        if not model.output_shape == (None, 5):
            t.failed = True
            t.msg = "Got a wrong output shape for the model"
            t.want = (None, 5)
            t.got = model.output_shape
        cases.append(t)

        t = test_case()
        if not (model.loss == 'sparse_categorical_crossentropy') |\
                (isinstance(model.loss, tf.losses.SparseCategoricalCrossentropy)): 
            t.failed = True
            t.msg = "Got a wrong loss"
            t.want = f"{tf.losses.SparseCategoricalCrossentropy} or 'sparse_categorical_crossentropy'"
            t.got = model.loss
        cases.append(t)

        t = test_case()
        if not 'accuracy' in list(hist.history.keys()):
            t.failed = True
            t.msg = "Got a wrong metric "
            t.want = "'accuracy' (there may be other metrics)"
            t.got = model.metrics_names[1:]
        cases.append(t)
        return cases

    cases = g()
    print_feedback(cases)

