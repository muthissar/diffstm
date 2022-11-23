from setuptools import setup
setup(
    python_requires='>=3.8.0',
    name='dstm',
    version='0.0.1',
    packages=['dstm'],
    install_requires=[
        'torch >= 1.8.0',
        'numpy >= 1.19.0',
        'scikit-learn >= 0.24.0',
        'pytorch-lightning >= 1.3.1',
        'matplotlib >= 3.3.0',
        # 'librosa >= 0.8.0',
        'pretty_midi >= 0.2.9',
        # 'test-tube >= 0.7.5 ',
        # 'nnAudio'
    ],
    # extras_require={
    #     'dev': [
    #         'tensorboard >= 2.2.0',
    #         'jupyter >= 1.0.0',
    #     ]
    # }
)
