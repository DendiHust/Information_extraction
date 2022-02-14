{
  "dataset_reader": {
    "type": "biaffine_reader",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": "./bert-base-chinese"
      }
    },
    "max_length": 512
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 1
    }
  },
  "train_data_path": "data/raw_data/tmp.json",
  "validation_data_path": "data/raw_data/tmp.json",
  "model": {
    "type": "biaffine",
    "dropout": 0.2,
    "label_num": 14,
    "entity_id_path": "./data/mid_data/biaffine_ent2id.json",
    "biaffine_dim": 128,
    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": "./bert-base-chinese"
        }
      }
    },
    "start_encoder":{
      "input_dim": 1536,
      "num_layers": 1,
      "hidden_dims": 128,
      "activations": "relu",
      "dropout": 0.0
    },
    "end_encoder":{
      "input_dim": 1536,
      "num_layers": 1,
      "hidden_dims": 128,
      "activations": "relu",
      "dropout": 0.0
    },
    "encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": 768,
      "dropout": 0.1,
      "bidirectional": true
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
    "callbacks": [
            {
                "type": "tensorboard",
                 "serialization_dir": "./summary"

             }
     ],
    "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["^__embedder*bias", "^__embedder*LayerNorm.bias", "^__embedder*LayerNorm.weight", "^__embedder*layer_norm.weight"], {"weight_decay": 0.0}],
              [["^__encoder*", "^__start*", "^__end*"],{"lr": 5e-4}]
            ]
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular"
        }
  }
}
