Sistem expert cu ontologii

O ontologie, în contextul informaticii și al inteligenței artificiale, este o reprezentare formală a cunoștințelor dintr-un anumit domeniu, structurată sub forma unui set de concepte, entități, proprietăți și relații între acestea. Scopul unei ontologii este de a permite înțelegerea comună și reutilizarea informației între sisteme și oameni, facilitând astfel interoperabilitatea și procesarea automată a datelor.

Ontologiile sunt utilizate în special în sisteme bazate pe cunoștințe, web semantic, baze de date, bioinformatică și multe alte domenii, oferind un cadru logic prin care se poate raționa asupra informației. Ele definesc nu doar ce entități există într-un anumit domeniu, ci și cum se leagă între ele, oferind o bază solidă pentru dezvoltarea de aplicații inteligente și coerente.

Programul de față utilizează o astfel de ontologie care implementeaza un sistem de diete. Ontologia este descrisă într-un fișier json

```json
{
"alimente": {
    "Mar": {
      "categorie": "Fructe",
      "nutrienti_la_100g": {
        "carbohidrati": 13.8,
        "proteine": 0.3,
        "grasimi": 0.2,
        "kcal": 57.0
      }
    },
    "Banana": {
      "categorie": "Fructe",
      "nutrienti_la_100g": {
        "carbohidrati": 22.8,
        "proteine": 1.1,
        "grasimi": 0.3,
        "kcal": 96.5
      }
    },
    "Linte": {
      "categorie": "Leguminoase",
      "nutrienti_la_100g": {
        "carbohidrati": 20.0,
        "proteine": 9.0,
        "grasimi": 0.4,
        "kcal": 121.6
      }
    }
  },
  "preferințe_dietetice": {
    "vegan": {
      "alimente": ["Mar", "Banana", "Morcov", "Cartof"]
    },
    "vegetarian": {
      "suitable_for": ["Mar", "Banana", "Morcov", "Cartof", "Iaurt natural", "Brânză telemea", "Ou"]
    },
  },
  "mese": {
    "mic_dejun": {
       "contine": ["Banana", "Iaurt natural", "Mar"]
    },
  },
  "condiții_medicale": {
    "diabet": {
      "descriere": "Se recomandă controlul strict al aportului de carbohidrati simpli.",
      "alimente_interzise": ["Banana", "Orez alb", "Pâine integrală"]
    },
  },
  "alergii": {
    "lactate": ["Brânză telemea", "Iaurt natural"],
    "nuci": ["Migdale"]
  }
}
```

Programul citește acest fișier și construiește un sistem expert care permite răspunsul la o serie de întrebări. 

```bash
Mancaruri pentru dieta vegana: {'alimente': ['Mar', 'Banana', 'Morcov', 'Cartof']}
Mancaruri pentru alergii la gluten: ['Pâine integrală']
Mancaruri cu mar: ['mic_dejun']
Nutrienti in carnea de pui: {'carbohidrati': 0.0, 'proteine': 31.0, 'grasimi': 3.6, 'kcal': 165.4}
    Informații nutritive:
    Calorii: 859.100 kcal
    Proteine: 91.900 g
    Carbohidrati: 87.300 g
    Grasimi: 11.800 g
```    


kcal = 4 × Carbohidrați + 4 × Proteine + 9 × Grăsimi