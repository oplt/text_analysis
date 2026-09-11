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


class AnnotationCampaignStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    RELEASED = "released"
    ARCHIVED = "archived"


class AnnotationMode(StrEnum):
    """Study design for an annotation campaign.

    ``blind_reliability`` hides peer codes, adjudications, model/LLM suggestions,
    and reliability results from annotators until their own coding is complete.
    ``ai_assisted`` may surface model suggestions.
    """

    BLIND_RELIABILITY = "blind_reliability"
    AI_ASSISTED = "ai_assisted"


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
    CORPUS_SYNTHESIS = "corpus_synthesis"
    INGESTION_QA = "ingestion_qa"
    DOCUMENT_CLEANING = "document_cleaning"
    SIMILARITY = "similarity"
    DUPLICATE_DETECTION = "duplicate_detection"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    READABILITY = "readability"
    STATISTICAL_MODEL = "statistical_model"
    MEASUREMENT_VALIDATION = "measurement_validation"
    DRIFT_MONITORING = "drift_monitoring"


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


class ModelLifecycleStatus(StrEnum):
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


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


class ResearchArtifactKind(StrEnum):
    """Discriminator for persisted research artifacts that wrap model outputs."""

    PREDICTION_SET = "prediction_set"


class ResearchMemoSourceType(StrEnum):
    """Origin of a research memo. Extensible without model redesign.

    Reserved for later (do not require schema change to add):
    ``topic_interpretation``, ``annotation_review``, ``classification_error_analysis``.
    """

    MANUAL = "manual"
    ASSISTANT_ANSWER = "assistant_answer"
    ANALYSIS_RESULT = "analysis_result"


class ResearchMemoStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
