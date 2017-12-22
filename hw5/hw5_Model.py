import numpy as np
from keras.layers import Input, Lambda, Embedding, Reshape, Merge, Dropout, Dense, Flatten, Dot
from keras.models import Sequential, Model
from keras import backend as K
from keras.layers.merge import concatenate, dot, add
from keras.regularizers import l2

def CFModel(MAX_USERID, MAX_MOVIEID, EMBEDDING_DIM):
    U_input = Input(shape = (1,))
    U = Embedding(MAX_USERID, EMBEDDING_DIM, embeddings_regularizer=l2(1e-5))(U_input)
    U = Reshape((EMBEDDING_DIM,))(U)
    U = Dropout(0.1)(U)

    V_input = Input(shape = (1,))
    V = Embedding(MAX_MOVIEID, EMBEDDING_DIM, embeddings_regularizer=l2(1e-5))(V_input)
    V = Reshape((EMBEDDING_DIM,))(V)
    V = Dropout(0.1)(V)

    U_bias = Embedding(MAX_USERID, 1, embeddings_regularizer=l2(1e-5))(U_input)
    U_bias = Reshape((1,))(U_bias)

    V_bias = Embedding(MAX_USERID, 1, embeddings_regularizer=l2(1e-5))(V_input)
    V_bias = Reshape((1,))(V_bias)

    out = dot([U,V], -1)
    out = add([out, U_bias, V_bias])

    out = Lambda(lambda x: x + K.constant(3.581712))(out)
    model = Model(inputs=[U_input, V_input], outputs=out)

    return model

def CFModel_report(MAX_USERID, MAX_MOVIEID, EMBEDDING_DIM):
    U_input = Input(shape = (1,))
    U = Embedding(MAX_USERID, EMBEDDING_DIM, embeddings_regularizer=l2(1e-5))(U_input)
    U = Reshape((EMBEDDING_DIM,))(U)
    U = Dropout(0.1)(U)

    V_input = Input(shape = (1,))
    V = Embedding(MAX_MOVIEID, EMBEDDING_DIM, embeddings_regularizer=l2(1e-5))(V_input)
    V = Reshape((EMBEDDING_DIM,))(V)
    V = Dropout(0.1)(V)

    # U_bias = Embedding(MAX_USERID, 1, embeddings_regularizer=l2(1e-5))(U_input)
    # U_bias = Reshape((1,))(U_bias)
    #
    # V_bias = Embedding(MAX_USERID, 1, embeddings_regularizer=l2(1e-5))(V_input)
    # V_bias = Reshape((1,))(V_bias)

    out = dot([U,V], -1)
    # out = add([out, U_bias, V_bias])

    out = Lambda(lambda x: x + K.constant(3.581712))(out)
    model = Model(inputs=[U_input, V_input], outputs=out)

    return model

def Deep_model(MAX_USERID, MAX_MOVIEID, EMBEDDING_DIM, dropout=0.1):
    u_input = Input(shape=(1,))
    u = Embedding(MAX_USERID, EMBEDDING_DIM)(u_input)
    u = Flatten()(u)

    m_input = Input(shape=(1,))
    m = Embedding(MAX_MOVIEID, EMBEDDING_DIM)(m_input)
    m = Flatten()(m)

    out = concatenate([u, m])
    out = Dropout(dropout)(out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(dropout)(out)
    out = Dense(128, activation='relu')(out)
    out = Dropout(dropout)(out)
    out = Dense(64, activation='relu')(out)
    out = Dropout(0.15)(out)
    out = Dense(EMBEDDING_DIM, activation='relu')(out)
    out = Dropout(0.2)(out)
    out = Dense(1, activation='relu')(out)

    model = Model(inputs=[u_input, m_input], outputs=out)
    return model

def rate(model, user_id, movie_id):
    return model.predict([np.array([user_id]), np.array([movie_id])])[0][0]
# class Model(Sequential):
#
#     def __init__(self, MAX_USERID, MAX_MOVIEID, EMBEDDING_DIM, **kwargs):
#         P = Sequential()
#         P.add(Embedding(MAX_USERID, EMBEDDING_DIM, input_length=1))
#         P.add(Reshape((EMBEDDING_DIM,)))
#         Q = Sequential()
#         Q.add(Embedding(MAX_MOVIEID, EMBEDDING_DIM, input_length=1))
#         Q.add(Reshape((EMBEDDING_DIM,)))
#         super(Model, self).__init__(**kwargs)
#         self.add(Merge([P, Q], mode='dot', dot_axes=1))
#
#     def rate(self, user_id, item_id):
#         return self.predict([np.array([user_id]), np.array([item_id])])[0][0]
