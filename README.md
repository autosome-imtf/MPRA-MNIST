## Для запуска .ipynb в examples помогает прописать такие строки

    clone https://github.com/autosome-imtf/MpraDataset.git
    cd MpraDataset
    pip install setuptools wheel
    python setup.py sdist bdist_wheel
    pip install -e .

Позднее планирую переписать инфо, выдваваемое о датасете и сделать его более удобочитаемым