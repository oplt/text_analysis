from enum import StrEnum


class UnitType(StrEnum):
    DOCUMENT = "document"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class AnnotationTaskStatus(StrEnum):
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DISAGREEMENT = "disagreement"
    ADJUDICATED = "adjudicated"


class AnnotationSource(StrEnum):
    ADJUDICATED_ONLY = "adjudicated_only"
    MAJORITY_VOTE = "majority_vote"
    SELECTED_ANNOTATOR = "selected_annotator"


class AnalysisRunType(StrEnum):
    PREPROCESSING = "preprocessing"
    FREQUENCY_ANALYSIS = "frequency_analysis"
    DFM = "dfm"
    NGRAM_ANALYSIS = "ngram_analysis"
    DICTIONARY_ANALYSIS = "dictionary_analysis"
    KEYNESS = "keyness"
    KWIC = "kwic"
    COOCCURRENCE = "cooccurrence"
    RELIABILITY = "reliability"
    TOPIC_MODEL = "topic_model"
    CLASSIFIER_TRAINING = "classifier_training"
    CLASSIFIER_PREDICTION = "classifier_prediction"
    ROBUSTNESS = "robustness"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    EXPORT = "export"
    CORPUS_STATS = "corpus_stats"
    SEGMENTATION = "segmentation"
    INGESTION_QA = "ingestion_qa"
    DOCUMENT_CLEANING = "document_cleaning"
    SIMILARITY = "similarity"
    DUPLICATE_DETECTION = "duplicate_detection"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    READABILITY = "readability"
    STATISTICAL_MODEL = "statistical_model"
    MEASUREMENT_VALIDATION = "measurement_validation"


class AnalysisRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClassifierTaskType(StrEnum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class ModelFamily(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_SVM = "linear_svm"
    MULTINOMIAL_NB = "multinomial_nb"
    COMPLEMENT_NB = "complement_nb"
    SGD_CLASSIFIER = "sgd_classifier"


class TopicAlgorithm(StrEnum):
    LDA = "lda"
    NMF = "nmf"


class LabelValue(StrEnum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class ProvenanceMode(StrEnum):
    HUMAN_ONLY = "human_only"
    MODEL_ONLY = "model_only"
    HUMAN_PREFERRED = "human_preferred"


class StratumMode(StrEnum):
    """How the target sample size is divided across strata."""

    PROPORTIONAL = "proportional"
    EQUAL = "equal"


class SamplingLevel(StrEnum):
    """Whether sampling selects individual units or whole documents."""

    UNIT = "unit"
    DOCUMENT = "document"
