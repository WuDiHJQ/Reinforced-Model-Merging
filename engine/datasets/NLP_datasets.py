from .utils import DatasetReader
from datasets import load_dataset


class QASC(DatasetReader):
    def __init__(self):

        super().__init__(dataset_stash=("qasc",), template_stash=("qasc",))

        self.name = "qasc"

        self.string_toLabelIdx = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
        }

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class WikiQA(DatasetReader):
    def __init__(self):

        super().__init__(dataset_stash=("wiki_qa",), template_stash=("wiki_qa",))

        self.name = "wiki_qa"

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class QuaRTz(DatasetReader):
    def __init__(self):

        super().__init__(dataset_stash=("quartz",), template_stash=("quartz",))

        self.name = "quartz"

        self.string_toLabelIdx = {"A": 0, "B": 1}

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class PAWS(DatasetReader):
    def __init__(self):

        super().__init__(
            dataset_stash=("paws", "labeled_final"),
            template_stash=("paws", "labeled_final"),
        )

        self.name = "paws"

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = example["label"]
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class StoryCloze(DatasetReader):
    def __init__(self):

        super().__init__(
            dataset_stash=("story_cloze", "2016"),
            template_stash=("story_cloze", "2016"),
        )

        self.name = "story_cloze"

    def _read_origin_dataset(self, split):

        # We use the test set of StoryCloze for validation and the validation set of StoryCloze
        # for train - following GPT3
        if split == "train":
            load_split = "validation"
        elif split == "validation":
            load_split = "test"

        if split not in self.cached_origData:
            # Do not use default method for loading dataset since the story_cloze dataset must be
            # downloaded manually and then we have to set data_dir to point to it.
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
                data_dir="../data",
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["answer_right_ending"]) - 1
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class Winogrande(DatasetReader):
    def __init__(self):

        super().__init__(
            dataset_stash=("winogrande", "winogrande_xl"),
            template_stash=("winogrande", "winogrande_xl"),
        )

        self.name = "winogrande"

    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["answer"]) - 1
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])


class WSC(DatasetReader):
    def __init__(self):

        super().__init__(
            dataset_stash=("super_glue", "wsc.fixed"),
            template_stash=("super_glue", "wsc.fixed"),
        )
        
        self.name = "wsc"
        
    def _read_origin_dataset(self, split):

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=split,
            )
            
            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):

        return super()._get_datasetTemplates([], ["Accuracy"])
