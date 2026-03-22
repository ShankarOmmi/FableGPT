class Config:

    # Training 

    batch_size = 8
    num_workers = 0

    train_split = 0.80
    test_split  = 0.10

    lr = 5e-6
    epochs = 100
    max_length = 512


    # Inference 

    top_k = 50
    temperature = 0.8   
    max_new_tokens = 200
    seed = 123
    context_length = 1024


config = Config()