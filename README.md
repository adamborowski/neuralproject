INSTRUCTIONS
============

1. run preprocess.py to read CSV files into binary format target/sentiments.p and target/tokens.p
    * manipulate set size by increasing limit variable
2. run build_network_input.py to transform string tokens into numbers understood by neural networks
3. run test.py to load prepared input and train model, then a prompt will be used for live demo