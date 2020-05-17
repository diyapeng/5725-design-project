import tensorflow as tf
import os
import numpy as np
from music21 import converter, instrument, note, chord, stream


# read training data's Notes
def get_notes():
    filepath = '/home/china/'
    files = os.listdir(filepath)
    Notes = []
    for file in files:
        try:
            stream = converter.parse(filepath + file)
            instru = instrument.partitionByInstrument(stream)
            if instru:  # if having instrument parts, read them
                notes = instru.parts[0].recurse()
            else:  # if there is no instrument, just take notes
                notes = stream.flat.notes
            for element in notes:
                if isinstance(element, note.Note):
                    Notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    Notes.append('.'.join(str(n) for n in element.normalOrder))
        except:
            pass
        with open('Note', 'a+')as f:
            f.write(str(Notes))
    return Notes


def get_model(inputs, notes_len, weights_file=None):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(512, input_shape=(inputs.shape[1], inputs.shape[2]),
                                   return_sequences=True))  
    model.add(tf.keras.layers.Dropout(0.3))  
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512))  
    model.add(tf.keras.layers.Dense(256))  
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(notes_len))  
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    if weights_file is not None:
        model.load_weights(weights_file)

    return model


# training model
def train():
    notes = get_notes()
    notes_len = len(set(notes))
    note_name = sorted(set(i for i in notes))  
    sequence_length = 100 
    note_dict = dict((j, i) for i, j in enumerate(note_name))  
    network_input = []  
    network_output = []  
    for i in range(0, len(notes) - sequence_length):
        # input 100, output 1
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_dict[k] for k in sequence_in])
        network_output.append(note_dict[sequence_out])
    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    normal_network_input = network_input / float(notes_len)  # normalization
    network_output = tf.keras.utils.to_categorical(network_output)  # output boolean matrix
    model = get_model(normal_network_input, notes_len)
    filepath = "weights-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',  # monitor object: loss
        verbose=0,
        save_best_only=True,
        mode='min'  
    )
    callbacks_list = [checkpoint]
    model.fit(normal_network_input, network_output, epochs=200, batch_size=256,
              callbacks=callbacks_list)  
    return network_input, normal_network_input, notes_len, note_name


# generate notes
def generate_notes(model, network_input, note_name, notes_len):
    randindex = np.random.randint(0, len(network_input) - 1)
    notedic = dict((i, j) for i, j in enumerate(note_name))  # convert the integer into chords
    pattern = list(network_input[randindex]) 
    predictions = []
    # randomly generate 100 chords
    for note_index in range(1000):
        # pattern = list(network_input[np.random.randint(0,500)])
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(notes_len)  
        prediction = model.predict(prediction_input, verbose=0)  
        index = np.argmax(prediction)
        # print(index)
        result = notedic[index]
        predictions.append(result)
        # move forward
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return predictions


# gernerate the music
def create_music():
    notes = get_notes()
    notes_len = len(set(notes))
    note_name = sorted(set(i for i in notes))
    sequence_length = 100  
    note_dict = dict((j, i) for i, j in enumerate(note_name))  
    network_input = []  
    network_output = []  
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_dict[k] for k in sequence_in])
        network_output.append(note_dict[sequence_out])
    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    normal_network_input = network_input / float(notes_len)  
    # print(len(network_input)) #1541019
    # network_input, normal_network_input,notes_len,note_name=train()
    files = os.listdir()
    minloss = {}
    for i in files:
        if 'weights' in i:
            num = i[11:15]
            minloss[num] = i
    best_weights = minloss[min(minloss.keys())]
    print('best model file:' + best_weights)
    model = get_model(normal_network_input, notes_len, best_weights)
    predictions = generate_notes(model, network_input, note_name, notes_len)
    offset = 0
    output_notes = []
    # generate Note or Chord Object
    for data in predictions:
        if ('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 1
    # generate music（Stream）
    midi_stream = stream.Stream(output_notes)
    # write into MIDI file
    midi_stream.write('midi', fp='output1.mid')


if __name__ == '__main__':
    with tf.device('/device:GPU:0'):
      train() # use for training
      create_music()
