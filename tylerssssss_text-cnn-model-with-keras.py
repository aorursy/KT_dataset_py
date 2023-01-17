from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, concatenate, Dropout
from keras.layers import Conv2D, MaxPooling2D, Embedding
from keras import regularizers

# sequence_length:    Expected max length of input sequence
# num_classes:        Number of category
# vocab_size:         The length of vocabulary
# embedding_size:     Dimensionality of character embedding (default: 128)
# filter_sizes:       Comma-separated filter sizes (default: '3,4,5')
# num_filters:        Number of filters per filter size (default: 128)
# dropout_keep_prob:  Dropout keep probability (default: 0.5)
# l2_reg_lambda:      L2 regularization lambda (default: 0.0)

def TextCNN(sequence_length=150, num_classes=2, vocab_size=2000, embedding_size=128, filter_sizes=[3,4,5], num_filters=100, dropout_keep_prob = 0.5, l2_reg_lambda=0.0):
    # input layer
    input_chars = Input(shape=(sequence_length,))
    
    # embedding layers
    embedded_chars = Embedding(vocab_size, embedding_size)(input_chars)
    embedded_chars_expanded = Reshape((sequence_length, embedding_size, 1))(embedded_chars)
    
    # conv layers
    convs = []
    for filter_size in filter_sizes:
        conv = Conv2D(num_filters, (filter_size, embedding_size), activation='relu')(embedded_chars_expanded)
        max_pool = MaxPooling2D((sequence_length - filter_size + 1, 1))(conv)
        convs.append(max_pool)
        
    # combine all the pooled features
    merge = concatenate(convs)
    
    # add dropout
    merge_dropout = Dropout(dropout_keep_prob)(merge)
    
    dense = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg_lambda))(merge_dropout)
    model = Model(input_chars, dense) 
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model
model = TextCNN()
model.summary()
