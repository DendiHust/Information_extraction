{
  "dataset_reader": {
    "type": "emr_ner",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "tokens",
        "lowercase_tokens": true
      }
    },
    "max_length": 512
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 10
    }
  },
  "train_data_path": "data/rel_data_ner_train_data_tag.json",
  "validation_data_path": "data/rel_data_ner_val_data_tag.json",
  "model": {
    "type": "crf_tragger",
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
      "type": "pytorch_transformer",
      "input_dim": 300,
      "num_layers":1,
      "feedforward_hidden_dim": 512,
      "num_attention_heads":4

    }
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
      "lr": 0.003
    }
  }
}
