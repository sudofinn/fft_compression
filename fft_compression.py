from PIL import Image
import numpy as np

img = Image.open("pics/alexpoatanpereira.png").convert("L") #dark-white with L
arr = np.asarray(img, dtype=np.float64)
#mach aus dem Bild eine 2D-Matrix, in der jeder Eintrag ein Pixelwert ist
#float64 konvertiert die pixel von 0-255 zu Gleitkommazahlen (höhere Genauigkeit, perfekt für FFT-Berechnung)

print(arr.shape, arr.dtype)#(900,1200) float64

F = np.fft.fft2(arr)
Fshift=np.fft.fftshift(F)
# Wenn man eine FFT berechnet ordnet NumPy die Frequenzen von 0 bis zur Nyquist-Frequenz erst nach rechts/unten
# und dann negativ wieder zurück, d.h. die niedrigen Frequezn sind oben links
# mit fftshift() sind die niedrigen Frequezen in der Mitte ( die wichtigen ) und die hohen Frequenzen
# ( die unwichtigen ) an den Rändern

print("FFT-Form:", Fshift.shape) #(900,1200)


# Bei der Kompression wollen wir die hohen Frequenzen ausblocken

#Jetzt kommt die Maske, die Kompression wo wir wir nur die zentralen ( niedrigen ) Frequenzen behalten

h,w = arr.shape
mask = np.zeros_like(Fshift) #Matrix voller 0 ( Größe 900x1200) , alle Einträge sind 0.0
#Das ist unsere Maske, eine Schablone die entscheidet, welche Frequenzen wir behalten und welche wir löschen

keep = int(0.1*min(h,w)) #Wir behalten die zentralen 10% der Frequenzen
#min(h,w) wählt die kleinste Bilddimension aus also hier 900

cy,cx = h//2 , w//2 #berechent die Koordinaten des Mittelpunktes des Frequenzbildes ( nach fftshift())
#nach fftshift() befinden sich die niedrigen Frequzen genau in der Mitte ( weil fftshift() nunma genau das veranlasst)
#und deshalb muss das Behaltequadrat um diesen Punkt herum platziert werden

mask[cy-keep:cy+keep, cx-keep:cx+keep] =1
# mask hatten wir oben ja schon definiert, es ist ein 2D-Array mit denselben Abmessungen wie das 
#Frequenzbild Fshift und ist momentan überall 0.0
#Das Ergebnis sieht dann etwa so aus 
#die 1 markieren die Frequenzen die behalten werden sollen ( niedrige Frequenzen )
#die 0 markieren die Frequenzen die gelöscht werden sollen ( die hohen Frequenzen )
'''
000000000000000000
000000011111000000
000000011111000000
000000011111000000
000000000000000000
'''
Fcompressed = Fshift * mask
# Frequenzbild * Filtermatrix ( punktweise Multiplikation von Numpy)
#Alle Frequenzen außerhalb der zentralen Region werden entfernt ( auf 0 gesetzt , da x*0 = 0)
#Das Bild wird weicher und kleiner im Informationsgehalt, also komprimiert

F_ishift = np.fft.ifftshift(Fcompressed) #Frequenzen werden wieder zurückverschoben
reconstructed = np.fft.ifft2(F_ishift).real # inverse FFT wird berechnet + Realteil

reconstructed_img = Image.fromarray(np.clip(reconstructed, 0,255).astype(np.uint8))
#np.clip(0,255) schneidet alle Werte außerhalb des Bereichs (0,255) ab
#uint8 wandelt den Datentyp von float64 in 8-Bit-Ganzzahlen um
reconstructed_img.save("pics/alexpoatanpereiracompressed.png")
