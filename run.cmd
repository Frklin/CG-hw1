.\bin\ycolorgrade --image tests\greg_zaal_artist_workshop.hdr --exposure 0 --output out\greg_zaal_artist_workshop_01.jpg
.\bin\ycolorgrade --image tests\greg_zaal_artist_workshop.hdr --exposure 1 --filmic --contrast 0.75 --saturation 0.75 --output out\greg_zaal_artist_workshop_02.jpg
.\bin\ycolorgrade --image tests\greg_zaal_artist_workshop.hdr --exposure 0.8 --contrast 0.6 --saturation 0.5 --grain 0.5 --output out\greg_zaal_artist_workshop_03.jpg

.\bin\ycolorgrade --image tests\toa_heftiba_people.jpg --exposure -1 --filmic --contrast 0.75 --saturation 0.3 --vignette 0.4 --output out\toa_heftiba_people_01.jpg
.\bin\ycolorgrade --image tests\toa_heftiba_people.jpg --exposure -0.5 --contrast 0.75 --saturation 0 --output out\toa_heftiba_people_02.jpg
.\bin\ycolorgrade --image tests\toa_heftiba_people.jpg --exposure -0.5 --contrast 0.6 --saturation 0.7 --tint-red 0.995 --tint-green 0.946 --tint-blue 0.829 --grain 0.3 --output out\toa_heftiba_people_03.jpg
.\bin\ycolorgrade --image tests\toa_heftiba_people.jpg --mosaic 16 --grid 16 --output out\toa_heftiba_people_04.jpg
.\bin\ycolorgrade --image tests\toa_heftiba_people.jpg --sigma-x 3 --sigma-y 3 --output out\toa_heftiba_people_06.jpg

.\bin\ycolorgrade --image tests\Old-Car.jpg --exposure -0.5 --filmic --sepia --output out\Old-Car.jpg
.\bin\ycolorgrade --image tests\Old-Car.jpg --sigma-x 5 --sigma-y 5 --output out\Old-Car_1.jpg 
.\bin\ycolorgrade --image tests\Keanu-Reeves-Is-The-Best.jpg --invert --output out\Keanu-Reeves-Is-The-Best_1.jpg 

.\bin\ycolorgrade --image tests\flower.jpg --laplace-alpha 0.25 --laplace-beta 1.0 --laplace-sigma 0.2 --output out\flower_1.jpg
.\bin\ycolorgrade --image tests\flower.jpg --laplace-alpha 0.25 --laplace-beta 1.0 --laplace-sigma 0.2 --graycale --output out\flower_2.jpg


.\bin\ycolorgrade --image tests\DSC_4311.JPG --laplace-alpha 0.25 --laplace-beta 1.0 --laplace-sigma 0.2 --output out\prova.JPG

