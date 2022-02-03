{
  "dataset_reader": {
    "type": "relation_base_reader",
    "config_file_path": "data/config/rel_label_info.json",
    "negative_sample_number":5,
    "use_pretrained_model":true,
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": "./bert-base-chinese",
        "tokenizer_kwargs": {
            "additional_special_tokens": [
                    "<S:CLI_COU>","</S:CLI_COU>","<O:CLI_COU>","</O:CLI_COU>",
                    "<S:SIGN>","</S:SIGN>","<O:SIGN>","</O:SIGN>",
                    "<S:LAB_RES>","</S:LAB_RES>","<O:LAB_RES>","</O:LAB_RES>",
                    "<S:LAB_ITEM>","</S:LAB_ITEM>","<O:LAB_ITEM>","</O:LAB_ITEM>",
                    "<S:EQU>","</S:EQU>","<O:EQU>","</O:EQU>",
                    "<S:NEG>","</S:NEG>","<O:NEG>","</O:NEG>",
                    "<S:ORG>","</S:ORG>","<O:ORG>","</O:ORG>",
                    "<S:DUR>","</S:DUR>","<O:DUR>","</O:DUR>",
                    "<S:PAST>","</S:PAST>","<O:PAST>","</O:PAST>",
                    "<S:TES_RES>","</S:TES_RES>","<O:TES_RES>","</O:TES_RES>",
                    "<S:TES>","</S:TES>","<O:TES>","</O:TES>",
                    "<S:DIS>","</S:DIS>","<O:DIS>","</O:DIS>",
                    "<S:SYM>","</S:SYM>","<O:SYM>","</O:SYM>",
                    "<S:DEG>","</S:DEG>","<O:DEG>","</O:DEG>",
                    "<S:FRE>","</S:FRE>","<O:FRE>","</O:FRE>"
            ]
        }
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
          "type": "pretrained_transformer",
          "model_name": "./bert-base-chinese",
          "tokenizer_kwargs": {

            "additional_special_tokens": [
                    "<S:CLI_COU>","</S:CLI_COU>","<O:CLI_COU>","</O:CLI_COU>",
                    "<S:SIGN>","</S:SIGN>","<O:SIGN>","</O:SIGN>",
                    "<S:LAB_RES>","</S:LAB_RES>","<O:LAB_RES>","</O:LAB_RES>",
                    "<S:LAB_ITEM>","</S:LAB_ITEM>","<O:LAB_ITEM>","</O:LAB_ITEM>",
                    "<S:EQU>","</S:EQU>","<O:EQU>","</O:EQU>",
                    "<S:NEG>","</S:NEG>","<O:NEG>","</O:NEG>",
                    "<S:ORG>","</S:ORG>","<O:ORG>","</O:ORG>",
                    "<S:DUR>","</S:DUR>","<O:DUR>","</O:DUR>",
                    "<S:PAST>","</S:PAST>","<O:PAST>","</O:PAST>",
                    "<S:TES_RES>","</S:TES_RES>","<O:TES_RES>","</O:TES_RES>",
                    "<S:TES>","</S:TES>","<O:TES>","</O:TES>",
                    "<S:DIS>","</S:DIS>","<O:DIS>","</O:DIS>",
                    "<S:SYM>","</S:SYM>","<O:SYM>","</O:SYM>",
                    "<S:DEG>","</S:DEG>","<O:DEG>","</O:DEG>",
                    "<S:FRE>","</S:FRE>","<O:FRE>","</O:FRE>"
            ]
        }
      }
    }
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
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["^_embedder*bias", "^_embedder*LayerNorm.bias", "^_embedder*LayerNorm.weight", "^_embedder*layer_norm.weight"], {"weight_decay": 0.0}],
              [["^_feed"],{"lr": 5e-4}]
            ]
        },

    "learning_rate_scheduler": {
        "type": "slanted_triangular"
    }
  }
}
