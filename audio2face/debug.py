import numpy as np
import pickle


def debug_deepspeech_results():
    audio_path = "../examples/audio_preprocessed/obama2.pkl"

    processed_audio = pickle.load(open(audio_path, 'rb'), encoding='iso-8859-1')
    # processed_audio = pickle.load(open(audio_path, 'rb'))


    print(type(processed_audio), processed_audio.shape)


def debug_audio2face_results():
    face_params_path = "../examples/test-result/obama2.npz"

    face_params = np.load(open(face_params_path, 'rb'))
    face_params = face_params['face']

    print(type(face_params), face_params.shape)
    print(np.min(face_params), np.max(face_params))


def debug_face_template():
    face_template_path = "../video_preprocess/train1_posenew.npz"
    # face_template_path = "../gangqiang_video_preprocess/gangqiang_posenew.npz"

    face_template = np.load(open(face_template_path, 'rb'))
    face_template = face_template['face']

    print(type(face_template), face_template.shape)

    print(np.min(face_template), np.max(face_template))

    std1 = np.std(face_template, axis=0)
    mean1 = np.mean(face_template, axis=0)

    print(std1.shape, mean1.shape)


def debug_deep3dface_results():
    matrix_npz = "../video_preprocess/train1_deep3Dface/train1.npz"

    face_template = np.load(open(matrix_npz, 'rb'))
    face_template = face_template['face']

    print(type(face_template), face_template.shape)

    print(np.min(face_template), np.max(face_template))



if __name__ == "__main__":
    # debug_deepspeech_results()

    # debug_audio2face_results()

    # debug_deep3dface_results()

    debug_face_template()