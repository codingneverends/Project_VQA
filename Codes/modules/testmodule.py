from yesorno import yesornno
from category import category
#qn = ["Is the lung healthy?","What modality used for this image?","Which part of the body does this image belong to?","Does the picture contain lung?"]
#print(yesornno(qn),category(qn))

def TestModule(qn):
    _category = category([qn])[0]
    _isopen = yesornno([qn])[0]
    return [_category,_isopen]