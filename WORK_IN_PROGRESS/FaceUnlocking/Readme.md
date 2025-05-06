
# Exemplu de recunoastere a fetei similar cu cel utilizat in dispozitivele mobile.

## Recunoasterea fetelor in dispozitivele mobile curente

### Apple (FaceID)

Face ID este un sistem de autentificare biometrică dezvoltat de Apple pentru iPhone și iPad Pro, care permite utilizatorilor să își deblocheze dispozitivele, să efectueze plăți și să acceseze date sensibile prin recunoașterea facială. Tehnologia se bazează pe o cameră TrueDepth, care proiectează și analizează peste 30.000 de puncte infraroșii pe fața utilizatorului, creând o hartă facială 3D unică. Această hartă este comparată cu datele stocate în dispozitiv pentru a verifica identitatea utilizatorului.

Un aspect important al Face ID este că învață și se adaptează la schimbările în aspectul utilizatorului, precum creșterea bărbii sau purtarea de machiaj, îmbunătățind astfel precizia și fiabilitatea recunoașterii. De asemenea, funcționează eficient în condiții de iluminare scăzută sau în întuneric complet, datorită iluminatorului infraroșu dedicat. 

#### Proiecția punctelor infraroșii
Proiectorul infraroșu din cadrul camerei TrueDepth proiectează o rețea de peste 30.000 de puncte infraroșii pe fața utilizatorului.
Aceste puncte sunt distribuite pe întreaga față, de la ochi până la bărbie, pentru a captura detalii fine ale trăsăturilor faciale.

#### Captarea imaginii cu senzorul infraroșu
După ce punctele infraroșii sunt proiectate pe față, camera infraroșie a sistemului TrueDepth captează aceste puncte.
Camera folosește un senzor specializat (de obicei un senzor VCSEL – Vertical-cavity surface-emitting laser) care măsoară distanța de la fiecare punct infraroșu până la fața utilizatorului.

#### Măsurarea adâncimii
Fiecare punct de lumină infraroșie este reflectat de fața utilizatorului și este capturat de senzor. În funcție de întârzierea cu care lumina se întoarce la senzor, se poate calcula distanța față de față (adică, adâncimea sau profilul tridimensional al feței).
Aceasta se bazează pe principiul triangulației: senzorul infraroșu și proiectorul emit lumină și măsoară unghiul la care lumina se întoarce după ce a lovit fața. Din aceste măsurători, sistemul poate determina distanța exactă la care se află fiecare punct reflectat, ceea ce formează o hartă tridimensională a feței.

#### Crearea unei hărți 3D a feței
În urma proiecției și măsurării adâncimii, sistemul creează o hartă 3D detaliată a feței utilizatorului. Aceasta include detalii despre adâncimea și conturul trăsăturilor faciale, precum pomeții, ochii, nasul, și bărbia.
În acest fel, chiar și trăsături subtile sunt capturate, iar fața este redată într-un model 3D precis.

#### Compararea cu datele stocate
Modelul 3D al feței utilizatorului este comparat cu unul salvat anterior în Secure Enclave (zona de stocare criptată a dispozitivului). Dacă hărțile se potrivesc, Face ID deblochează dispozitivul.

### Android

Majoritatea dispozitivelor Android se bazeaza pe sisteme de recunoastere a fetelor din imagini. Exista cateva modele care utilizeaza senzori infrarosu (mai mult sau mai putin similari cu IPhone). 


## Recunoasterea faciala cu ajutorul mediapipe.

Acest tip de recunoastere faciala permite inregistrarea catorva cadre ale unei fete urmand ca pe baza acestora aceeasi persoana sa fie recunoscuta mai tarziu. 
Inregistrarea implica preluarea a 5 cadre din camera web urmata de extragerea cu ajutorul mediapipe a unor repere faciale si inregistrarea lor intr-un fisier json. 

Ulterior, procesul de recunoastere preia frame-urile de la webcam, identifica fata in cadre si extrage reperele din aceste fete. Toate aceste colectii de repere sunt comparate cu reperele stocate. 

Comparatia se face utilizand distanta euclidiana dintre repere - fiecare lista de repere este privita ca un punct intr-un spatiu cu 468 de dimensiuni. Fiecare cadru din webcam din timpul recunoasterii se compara cu ajutorul acestei distante cu cele stocate in json. Daca distanta este mai mica de 0.6 atunci fata este recunoscuta. 

# Rulare

Programul nu poate rula in WSL, trebuie rulat direct din Windows. Mediapipe nu este disponibil pentru versiunea de Python instalata in mod obisnuit pe Windows (3.13). Trebuie instalata o versiune mai veche si activat un mediu virtual. 

Windows virtual env
C:\Users\smilutinovici\AppData\Local\Programs\Python\Python312\python.exe -m venv mediapipe_env

mediapipe_env\Scripts\activate

