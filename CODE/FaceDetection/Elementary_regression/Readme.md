## Functia de cost

Această funcție de cost (YOLOFaceLoss) este concepută pentru a antrena un model de tip YOLO adaptat pentru detectarea fețelor într-o imagine, unde fiecare celulă a unei grile bidimensionale decide dacă există o față și estimează poziția și dimensiunile acesteia. Funcția ia în considerare atât celulele care conțin fețe, cât și cele care nu conțin, aplicând penalizări diferite în funcție de situație.

Funcția primește ca intrare două tensori: pred (predicțiile modelului) și target (valorile reale), fiecare de dimensiune (batch, grid_size, grid_size, 5), unde cei cinci coeficienți reprezintă: probabilitatea prezenței unei fețe, coordonatele centrului (x, y) și dimensiunile (lățime, înălțime) ale feței în cadrul celulei.

Se definesc două tipuri de măști:

 - obj_mask identifică celulele unde o față este prezentă (valoarea probabilității reale este mai mare decât zero).
 - no_obj_mask identifică celulele unde nu există față (valoarea probabilității reale este zero).

Pentru celulele care conțin fețe, se calculează:

 - pierdere de coordonate: penalizează diferențele dintre predicțiile și valorile reale pentru x, y, lățime și înălțime, folosind MSE (Mean Squared Error), multiplicat cu un factor lambda_coord, ce controlează cât de mult contează acest termen în totalul pierderii.
 - pierdere pentru scorul de încredere (obj_loss): evaluează cât de bine estimează modelul probabilitatea că o față este prezentă într-o celulă. Se folosește Binary Cross Entropy (BCE) între valoarea prezisă și cea reală.

Pentru celulele care nu conțin fețe, se calculează:

pierdere pentru absența obiectului (no_obj_loss): penalizează predicțiile false pozitive (adică atunci când modelul prezice o față acolo unde nu este). Se aplică BCE, ponderat cu un coeficient lambda_noobj, care este în mod normal foarte mic pentru a nu domina pierderea totală.

La final, funcția adună cele trei componente de pierdere și returnează pierderea totală împreună cu fiecare componentă individuală, oferind astfel o imagine detaliată asupra contribuției fiecărui tip de eroare. Este implementat și un mecanism de detectare a valorilor NaN, pentru a identifica eventualele probleme numerice în timpul antrenării.