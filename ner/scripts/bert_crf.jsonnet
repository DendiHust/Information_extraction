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
  "train_data_path": "data/raw_data/tmp.json",
  "validation_data_path": "data/raw_data/dev.json",
  "model": {
    "type": "bert_crf",
    "bert_path": "./bert-base-chinese",
    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": "./bert-base-chinese"
        }
      }
    }
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 1,
    "grad_clipping": 5.0,
    "validation_metric": "+f1-measure-overall",
    "checkpointer": {
        "keep_most_recent_by_count": 3
    },
    "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}]
            ]
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular"
        }
  }
}
