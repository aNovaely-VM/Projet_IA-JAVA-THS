# Projet_IA-JAVA-THS


/*

Compiler & démarrer le programme :
    javac -d Class neurone/*.java Image.java Compilation.java
    java -cp Class Compilation

Le meilleur mode en ratio temps/précision est :
    2 ReLu
    * (GRI est obligatoire)
    4 HOG
    1/2 --> meilleure réussite pour 1 seul neuronne (~95%), mais plus "utile" avec 3 (~89%)
    2 shuffle
    !! Un grand nombre (1000+) d'itération est facilement faisable grâce à la faible durée, mais je recommande un 200, c'est suffisant !!

*/