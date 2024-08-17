import os
import copy
import torch
import random
from torch.utils import data
from datasets import load_dataset
from promptsource.templates import DatasetTemplates


class DatasetReader(object):
    """
    DatasetReader objects reads dataset and has all attributes specific to dataset
    """

    def __init__(self, dataset_stash, template_stash):

        self.dataset_stash = dataset_stash
        self.template_stash = template_stash

        self.all_templates = self._get_datasetTemplates(None, None)

        self.cached_origData = {}
        self.cached_datasets = {}

    def _get_origData(self, split):
        return self._read_origin_dataset(split)

    def _read_origin_dataset(self, split):
        """
        Reads the original dataset split from huggingface. Converts the label to an int and returns the updated dataset.
        Args:
            split:

        Returns:

        """
        load_split = "validation" if split == "test" else split
        load_split = "validation" if load_split == "validation_full" else load_split

        if split not in self.cached_origData:
            print(f"Loading Full Data for {self.name}")
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=load_split,
            )
            orig_data = []
            # converting label to int and caching the split of the dataset.
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        """
        Returns a list of all templates for the dataset with the given metrics and not in the list of templates to ignore.
        Args:
            templateNames_toIgnore:
            metrics_toUse: specify the metric to use so that we only include templates which
                           match the metric we want to use

        Returns:

        """
        all_templates = []

        # Get original templates from promptsource
        for template in DatasetTemplates(*self.template_stash).templates.values():
            # Filter out templates that
            # 1) are not designed for original task
            # 2) have different metrics than we want to use
            # 3) are ones that we want to ignore based on the name
            if template.metadata.original_task:
                should_ignoreTemplate = False

                for metric in template.metadata.metrics:
                    if metric not in metrics_toUse:
                        should_ignoreTemplate = True

                for template_name in templateNames_toIgnore:
                    if template.name == template_name:
                        should_ignoreTemplate = True

                if not should_ignoreTemplate:
                    all_templates.append(template)
        return all_templates

    def _applyTemplate_toData(
        self, orig_data, num_templates, template_idx, is_evaluation
    ):
        """
        Args:
            orig_data:
            num_templates:
            template_idx:
            is_evaluation:

        Returns:

        """
        dataset = []

        for datapoint_idx, datapoint in enumerate(orig_data):

            # Use fixed template across entire dataset
            if template_idx >= 0:
                templateIdx_forDatapoint = template_idx

            # Use all templates across entire dataset, where different datapoints can get
            # different templates. However, a datapoint is always matched with the same template
            elif template_idx == -1:
                templateIdx_forDatapoint = datapoint_idx % num_templates

            # select a random template for the example.
            elif template_idx == -3:
                templateIdx_forDatapoint = random.randint(0, len(self.all_templates))

            else:
                raise ValueError(f"Invalid template index {templateIdx_forDatapoint}")

            template = self.all_templates[templateIdx_forDatapoint]
            new_datapoint = copy.deepcopy(datapoint)

            # For evaluation, we add the answer_choices if they exist
            # if is_evaluation:
            #     answer_choices = template.get_answer_choices_list(datapoint)
            #     if answer_choices is not None:
            #         new_datapoint["answer_choices"] = answer_choices
            answer_choices = template.get_answer_choices_list(datapoint)
            if answer_choices is not None:
                new_datapoint["answer_choices"] = answer_choices

            # We apply the template to datapoint instead of new_datapoint since the answer_choices
            # are added in the template function, and so applying the template to new_datapoint
            # will cause an error with the answer_choices key
            input_txt, target_txt = template.apply(datapoint)
            new_datapoint["input"] = input_txt

            # For non-evaluation or tasks where they are no answer_choices, we just add the target (
            # the correct answer_choice)
            # if not is_evaluation or "answer_choices" not in new_datapoint:
            #     new_datapoint["target"] = target_txt
            new_datapoint["target"] = target_txt
            
            dataset.append(new_datapoint)

        return dataset

    def get_dataset(
        self, split, template_idx, is_evaluation, max_samples_per_dataset=None
    ):
        """
        Create dataset that includes the template

        Args:
            split:
            template_idx:
                if >=0, then we use the fixed template_idx across entire dataset
                if ==-1, then we use all template across entire the dataset, where different
                         datapoints can have different templates. A datapoint will always be
                         mapped to the same template though
                if ==-2, then we take the cross product of all templates and all datapoints.
                if ==-3, apply a random template to each datapoint.
            is_evaluation: whether the split is for evaluation (where it will have answer_choices)
                            or for training (where it will only have the target)
        Returns:
            dataset:
        """
        if (split, template_idx) not in self.cached_datasets:
            orig_data = self._get_origData(split)
            total_examples = len(orig_data)
            print(
                f"Dataset:{self.name.upper()}\tSplit:{split}\tSelected Examples: {len(orig_data)}\tNum Total Example:{total_examples}"
            )
            num_templates = self.get_numTemplates()

            # template_idx -2 means we do a cross product of each datapoint with each template
            if template_idx == -2:
                dataset = []
                for iterate_templateIdx in range(num_templates):
                    dataset.extend(
                        self._applyTemplate_toData(
                            orig_data, num_templates, iterate_templateIdx, is_evaluation
                        )
                    )
            # otherwise apply template to dataset
            else:
                dataset = self._applyTemplate_toData(
                    orig_data, num_templates, template_idx, is_evaluation
                )
            # shuffle examples and select max_samples perdataset as same examples with different templates will occur together.
            random.Random(4).shuffle(dataset)
            total_examples_with_templates = len(dataset)
            dataset = (
                dataset[:max_samples_per_dataset]
                if max_samples_per_dataset
                else dataset
            )
            print(
                f"Dataset:{self.name.upper()}\tSplit:{split}\tNum Selected Example with Templates:{len(dataset)}\tTemplate Idx:{template_idx}\tNum Templates:{num_templates}\tNum Examples with Template:{total_examples_with_templates}"
            )

            self.cached_datasets[(split, template_idx)] = dataset

        return self.cached_datasets[(split, template_idx)]

    def get_numTemplates(self):
        return len(self.all_templates)

    def get_metricsForDataset(self):
        return self.all_templates[0].metadata.metrics


class DatasetWrapper(data.Dataset):

    def __init__(self, dataset, tokenizer, device):

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, get_idx):

        datapoint = self.dataset[get_idx]
        input_dict = self.tokenizer(
            datapoint["input"], return_tensors="pt", truncation=True
        )
        input_ids = input_dict["input_ids"][0]
        input_mask = input_dict["attention_mask"][0]

        allChoices_ids = []
        allChoices_masks = []

        new_datapoint = copy.deepcopy(datapoint)

        new_datapoint.update(
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
            }
        )

        if "answer_choices" in datapoint:
            for choice in datapoint["answer_choices"]:
                # This assumes tokenizer does not add BOS token, which is true for T5
                choice_dict = self.tokenizer(
                    choice, return_tensors="pt", truncation=True
                )
                allChoices_ids.append(choice_dict["input_ids"][0])
                allChoices_masks.append(choice_dict["attention_mask"][0])

            new_datapoint.update(
                {
                    "all_choices_ids": allChoices_ids,
                    "all_choices_mask": allChoices_masks,
                }
            )
        if "target" in datapoint:
            target_dict = self.tokenizer(
                datapoint["target"], return_tensors="pt", truncation=True
            )
            target_ids = target_dict["input_ids"][0]
            target_mask = target_dict["attention_mask"][0]

            new_datapoint.update(
                {
                    "target_ids": target_ids,
                    "target_mask": target_mask,
                }
            )

        return new_datapoint

    def collate_fn(self, batch_ofDatapoints):
        """
        Convert a batch of datapoints into a datapoint that is batched.  This is meant to
        override the default collate function in pytorch.

        Args:
            batch_ofDatapoints:

        Returns:

        """
        datapoint_batched = {}

        for datapoint in batch_ofDatapoints:
            for (k, v) in datapoint.items():
                if k in datapoint_batched:
                    # Each value in all_choices is already a list, so we extend and not append.
                    if "all_choices" in k:
                        datapoint_batched[k].extend(v)
                    else:
                        datapoint_batched[k].append(v)
                else:
                    # Each value in all_choices is already a list, so we do not need to
                    # initialize a list with v in it, and can just use v.
                    if "all_choices" in k:
                        datapoint_batched[k] = v
                    else:
                        datapoint_batched[k] = [v]

        # Pad ids and mask to maximum length in batch
        for (k, batch_ofValues) in datapoint_batched.items():
            # If id or mask is in key, this means we need to pad to the longest sequence length
            if ("ids" in k) or ("mask" in k):
                if "ids" in k:
                    padToken_id = self.tokenizer.pad_token_id
                    if padToken_id is None:
                        padToken_id = self.tokenizer.eos_token_id
                elif "mask" in k:
                    padToken_id = 0
                else:
                    raise ValueError(
                        f"The key {k} has ids or masks but is not recognized"
                    )
                datapoint_batched[k] = torch.nn.utils.rnn.pad_sequence(
                    batch_ofValues, batch_first=True, padding_value=padToken_id
                )

                if self.device is not None:
                    datapoint_batched[k] = datapoint_batched[k].to(self.device)

            elif k == "lbl":
                datapoint_batched[k] = torch.tensor(batch_ofValues)

                if self.device is not None:
                    datapoint_batched[k] = datapoint_batched[k].to(self.device)

        return datapoint_batched
