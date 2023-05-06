from flask import Flask, redirect, url_for, request, render_template
from flask_cors import CORS, cross_origin
from yesorno import yesornno
from category import category
from closeclassifier import CloseClassifier
from openclassifier import OpenClassifier
app = Flask(__name__)
CORS(app)

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/getresult',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      image = request.files['image']
      path = 'tmp/'+image.filename
      image.save(path)
      qn = request.form['qn']
      print("\n\nRecived Request\n\nQuestion : ",qn)
      _isopen = yesornno([qn])[0]
      print("\nQuestion type : ",_isopen)
      _category = category([qn])[0]
      print("\nCategory : ",_category)
      if _isopen == "OPEN":
         result = OpenClassifier(path,qn,_category)
      else:
         result = CloseClassifier(path,qn,_category)
      print("\nResult : ",result,"\n")
      return {"status":"OK","content_type":_category,"question_type":_isopen,"answer":result} , 200

@app.route('/')
def home():
   return render_template('index.html')
if __name__ == '__main__':
   app.run(host='localhost', port=1212)