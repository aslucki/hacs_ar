from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D, Activation, add, GlobalAveragePooling3D

def get_C3D_base(inputs):

    conv1 = Conv3D(64, (3, 3, 3), activation='relu',
                   padding='same', name='conv1')(inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                         padding='valid', name='pool1')(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu',
                   padding='same', name='conv2')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool2')(conv2)

    conv3a = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3a')(pool2)
    conv3a = BatchNormalization()(conv3a)
    conv3b = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3b')(conv3a)
    conv3b = BatchNormalization()(conv3b)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool3')(conv3b)

    conv4a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4a')(pool3)
    conv4a = BatchNormalization()(conv4a)
    conv4b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4b')(conv4a)
    conv4b = BatchNormalization()(conv4b)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool4')(conv4b)

    conv5a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5a')(pool4)
    conv5a = BatchNormalization()(conv5a)
    conv5b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5b')(conv5a)
    conv5b = BatchNormalization()(conv5b)
    zeropad5 = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)),
                             name='zeropad5')(conv5b)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool5')(zeropad5)

    return pool5


def original_c3d(use_negative_samples=True):
    inputs = Input(shape=(16, 112, 112, 3,))
    base = get_C3D_base(inputs)

    flattened = Flatten()(base)
    fc6 = Dense(4096, activation='relu', name='fc6')(flattened)
    #dropout1 = Dropout(rate=0.3)(fc6)

    fc7 = Dense(4096, activation='relu', name='fc7')(fc6)
    #dropout2 = Dropout(rate=0.2)(fc7)

    predictions = Dense(200, activation='softmax', name='predictions')(fc7)

    if use_negative_samples:
        label_predictions = Dense(1, activation='sigmoid', name='negative_label')(fc7)
        predictions = [predictions, label_predictions]

    return Model(inputs=inputs, outputs=predictions)


def resnet_c3d(use_negative_samples=True):
    inputs = Input(shape=(16, 112, 112, 3,))
    base = get_resnet3D_base(inputs)

    predictions = Dense(200, activation='softmax', name='fc1000')(base)
    if use_negative_samples:
        label_predictions = Dense(1, activation='sigmoid', name='negative_label')(base)
        predictions = [predictions, label_predictions]

    return Model(inputs=inputs, outputs=predictions)


def c3d_word_embedding(use_negative_samples=True):
    inputs = Input(shape=(16, 112, 112, 3,))
    base = get_C3D_base(inputs)

    flattened = Flatten()(base)
    fc6 = Dense(5096, activation='relu', name='fc6')(flattened)
    dropout1 = Dropout(rate=0.1)(fc6)

    fc7 = Dense(2048, activation='relu', name='fc7')(dropout1)
    dropout2 = Dropout(rate=0.1)(fc7)

    predictions = Dense(300, activation='sigmoid', name='fc8')(dropout2)

    outputs = predictions
    if use_negative_samples:
        label_predictions = Dense(1, activation='sigmoid', name='negative_label')(dropout2)
        outputs = [predictions, label_predictions]

    return Model(inputs=inputs, outputs=outputs)


def get_model(name, use_negative_samples):
    if name == 'original':
        return original_c3d(use_negative_samples=use_negative_samples)
    elif name == 'word_embeddings':
        return c3d_word_embedding(use_negative_samples=use_negative_samples)
    elif name == 'resnet_3d':
        return resnet_c3d(use_negative_samples=use_negative_samples)


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(filters1, (1, 1, 1),
                               kernel_initializer='he_normal',
                               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters2, kernel_size,
                               padding='same',
                               kernel_initializer='he_normal',
                               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters3, (1, 1, 1),
                               kernel_initializer='he_normal',
                               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(filters1, (1, 1, 1), strides=strides,
                               kernel_initializer='he_normal',
                               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters2, kernel_size, padding='same',
                               kernel_initializer='he_normal',
                               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv3D(filters3, (1, 1, 1),
                               kernel_initializer='he_normal',
                               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv3D(filters3, (1, 1, 1), strides=strides,
                                      kernel_initializer='he_normal',
                                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def get_resnet3D_base(inputs):
    bn_axis = 3
    x = ZeroPadding3D(padding=(3, 3, 3), name='conv1_pad')(inputs)
    x = Conv3D(64, (7, 7, 7),
                               strides=(2, 2, 2),
                               padding='valid',
                               kernel_initializer='he_normal',
                               name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding3D(padding=(1, 1, 1), name='pool1_pad')(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling3D(name='avg_pool')(x)

    return x