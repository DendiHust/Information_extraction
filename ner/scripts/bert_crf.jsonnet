{
  "dataset_reader": {
    "type": "tag_reader",

    "max_length": 512
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 1
    }
  },
  "train_data_path": "data/raw_data/train.json",
  "validation_data_path": "data/raw_data/dev.json",
  "model": {
    "type": "bert_crf",
    "bert_path": "./bert-base-chinese"
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "-loss",
    "checkpointer": {
        "keep_most_recent_by_count": 3
    },
    "optimizer": {
      "type": "adam",
      "lr": 0.0003
    }
  }
}
