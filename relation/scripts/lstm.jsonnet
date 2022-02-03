{
  "dataset_reader": {
    "type": "relation_base_reader",
    "config_file_path": "data/config/rel_label_info.json",
    "negative_sample_number":5,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "tokens",
        "lowercase_tokens": true
      }
    },
    "max_length": 512
  },
  "train_data_path": "data/raw/rel_train_data.json",
  "validation_data_path": "data/raw/rel_val_data.json",

  "model": {
    "type": "relation_base_model",
    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "trainable": true
        }
      }
    },

    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 256,
      "dropout": 0.5,
      "bidirectional": true
    },
    "loss_func": "focal_loss"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 5
    }
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 1,
    "grad_clipping": 5.0,
    "validation_metric": "-loss",
    "checkpointer": {
        "keep_most_recent_by_count": 3
    },
    "callbacks": [
            {
                "type": "tensorboard",
                 "serialization_dir": "./summary"

             }
     ],
    "optimizer": {
      "type": "adam",
      "lr": 0.003
    },

    "learning_rate_scheduler": {
        "type": "slanted_triangular"
    }
  }
}