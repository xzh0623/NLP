    # try:
    #     with np.load(file_path, allow_pickle=True) as data:
    #         # matrix = data['features_matrix'].item()  # Change from 'arr_0' to 'features_matrix'
    #         matrix = data['features_matrix'].item()
    #         print(f"- {feature_name}")
    #         print("Shape of Matrix:", matrix.shape)
    #         print("Non-zero elements in Matrix:", matrix.nnz)
    # except FileNotFoundError:
    #     print(f"File not found: {file_path}")
    # except KeyError:
    #     print(f"Key error in {file_path}: ensure the correct key is being accessed.")
    # except Exception as e:
    #     print(f"An error occurred while loading {file_path}: {str(e)}")