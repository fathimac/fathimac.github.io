# Guide for using Python CGI for collecting annotation

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)  
Date: 2020/11/21


### Setup:

* The folder AnnoImSelect provides a simple example: 
* Main annotation files, either:
	- For one image at at time: ./cgi-bin/annotate.cgi (extremely simple 120 lines of code)
	- For multiple images at once: ./cgi-bin/annotate_multi.cgi and ./cgi-bin/save.cgi (extremely simple 150 lines of code)
	
* IMPORTANT: Modify the path to your python bin on the first line of *.cgi scripts: `#!/Path/to/your/envs/python3`


### Execute:

* Suppose the images are in `/Users/hoai/tmp/DayMoon`


``` bash 
cd AnnoImSelect/
ln -s /Users/hoai/tmp/DayMoon imDir # create a symbolic link
cd imDir
ls *.{jpg,jpeg,png,bmp,JPG,PNG,BMP} > ids.txt % or provide a list of filenames in imDir/ids.txt
cd ..

python3 -m http.server --cgi 8000
# python2 -m CGIHTTPServer 8000

```

### Try the examples
* On your browser:

	- annotate one image at a time: [http://localhost:8000/cgi-bin/annotate.cgi?id=0&dir=imDir](http://localhost:8000/cgi-bin/annotate.cgi?id=0&dir=imDir)
	- annotate multiple images at once: [http://localhost:8000/cgi-bin/annotate_multi.cgi?dir=imDir&page=0](http://localhost:8000/cgi-bin/annotate_multi.cgi?dir=imDir&page=0)

* The annotation will be stored in a database file: `imDir/anno.db`
* To see the annotation, run: `python show_anno.py imDir/anno.db`

### NOTE. READ CAREFULLY:

* Before you start new annotation, you might want to delete the old annotation file imDir/anno.db
* Both of the above examples use the same annotation file, so they might override each other. Be careful. Use one! 




