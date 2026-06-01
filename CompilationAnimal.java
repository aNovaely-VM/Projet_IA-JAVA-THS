import java.util.*;

/**
 * Programme principal de classification chats / chiens.
 *
 * Chaîne de traitement :
 *   1. Lecture des images d'entraînement (train/cat + train/dog)
 *   2. Labellisation automatique selon le nom du dossier
 *   3. Normalisation des pixels (valeurs ramenées dans [0,1])
 *   4. Mélange aléatoire des données
 *   5. Entraînement du neurone
 *   6. Lecture + test sur les images de test
 *   7. Affichage du taux de précision
 *
 * Compiler et lancer :
 *   javac neurone/*.java Image.java CompilationAnimal.java
 *   java -cp ".;neurone" CompilationAnimal
 */
public class CompilationAnimal
{
	// =========================================================
	// CONFIGURATION — modifiez ces valeurs pour vos expériences
	// =========================================================

	/** Chemin vers le dossier d'entraînement contenant les sous-dossiers cat/ et dog/ */
	static final String DOSSIER_TRAIN = "dataset_groupe_6/train/";

	/** Chemin vers le dossier de test contenant les sous-dossiers cat/ et dog/ */
	static final String DOSSIER_TEST  = "dataset_groupe_6/test/";

	/**
	 * MSE (Mean Squared Error) cible pour arrêter l'apprentissage.
	 * Plus cette valeur est basse, plus le neurone doit être précis avant de s'arrêter,
	 * mais plus l'apprentissage sera long. Avec Heaviside, cette valeur peut ne jamais
	 * être atteinte (boucle arrêtée par MAX_ITERATIONS dans Neurone.java).
	 */
	static final float MSE_LIMITE = 0.05f;

	/**
	 * Taille de redimensionnement des images en pixels (IMAGE_SIZE x IMAGE_SIZE).
	 * Toutes les images du dataset sont réduites à cette taille par sous-échantillonnage.
	 * Le neurone aura IMAGE_SIZE² entrées (ex: 64x64 = 4096 entrées).
	 * Une valeur plus grande = plus d'information mais apprentissage plus lent.
	 */
	static final int IMAGE_SIZE = 64;

	/**
	 * Mode de lecture des images.
	 * true  = niveaux de gris : 1 valeur par pixel (luminosité), IMAGE_SIZE² entrées.
	 * false = couleur RGB     : 3 valeurs par pixel (rouge, vert, bleu), 3*IMAGE_SIZE² entrées.
	 */
	static final boolean NIVEAUX_DE_GRIS = false;

	/**
	 * Type de fonction d'activation du neurone.
	 * "heavyside" : sortie binaire (0 ou 1), apprentissage par sauts, relativement lent.
	 * "relu"      : sortie = max(0, x), rapide mais peut diverger sur grandes entrées et arrive vite à sa limite.
	 * "sigmoide"  : sortie continue (0.0 à 1.0), convergence rapide puis douce, la plus efficace.
	 */
	static final String TYPE_NEURONE = "sigmoide";


	// =========================================================

	public static void main(String[] args)
	{
		System.out.println("=== Classification chats / chiens ===\n");
		System.out.println("Neurone : " + TYPE_NEURONE);
		System.out.println("Image   : " + IMAGE_SIZE + "x" + IMAGE_SIZE
			+ (NIVEAUX_DE_GRIS ? " (niveaux de gris)" : " (RGB)"));
		System.out.println("MSE limite : " + MSE_LIMITE + "\n");

		// ----------------------------------------------------------
		// 1. Chargement des images d'entraînement
		//    On lit tous les fichiers image du dossier train/,
		//    on les labellise selon leur sous-dossier (cat ou dog),
		//    et on les réduit à IMAGE_SIZE x IMAGE_SIZE pixels.
		// ----------------------------------------------------------
		System.out.println("--- Chargement des données d'entraînement ---");
		List<Image> imagesTrain = chargerImages(DOSSIER_TRAIN);
		if (imagesTrain == null || imagesTrain.isEmpty()) {
			System.err.println("Aucune image trouvée dans " + DOSSIER_TRAIN);
			return;
		}
		System.out.println(imagesTrain.size() + " images chargées.\n");

		// ----------------------------------------------------------
		// 2. Conversion en tableaux de flottants + normalisation
		//    Le neurone ne manipule que des float[], pas des objets Image.
		//    - entreesTrain[i] = tableau des pixels normalisés de l'image i
		//    - labelsTrain[i]  = 1.0 si chat, 0.0 si chien
		//    La normalisation (division par 255) ramène les pixels de [0-255]
		//    à [0.0-1.0], ce qui stabilise et accélère l'apprentissage.
		// ----------------------------------------------------------
		int nbEntrees = imagesTrain.get(0).taille(); // IMAGE_SIZE² ou 3*IMAGE_SIZE² si RGB
		float[][] entreesTrain = new float[imagesTrain.size()][nbEntrees];
		float[]   labelsTrain  = new float[imagesTrain.size()];
		for (int i = 0; i < imagesTrain.size(); ++i) {
			Image img = imagesTrain.get(i);
			int[] donnees = img.donnees();
			for (int j = 0; j < nbEntrees; ++j)
				entreesTrain[i][j] = donnees[j] / 255.0f; // normalisation [0-255] -> [0.0-1.0]
			// label : 1 = chat (neurone actif), 0 = chien (neurone inactif)
			labelsTrain[i] = (img.label() == Image.LabelChat) ? 1.0f : 0.0f;
		}

		// ----------------------------------------------------------
		// 3. Mélange aléatoire des données avant apprentissage
		//    Sans mélange, le neurone verrait d'abord tous les chats
		//    puis tous les chiens, ce qui biaiserait l'apprentissage.
		//    Le mélange garantit une présentation équilibrée à chaque epoch.
		// ----------------------------------------------------------
		melanger(entreesTrain, labelsTrain);
		System.out.println("Données mélangées.\n");

		// ----------------------------------------------------------
		// 4. Création du neurone
		//    Le neurone est créé avec nbEntrees entrées (= nb de pixels).
		//    Ses poids sont initialisés aléatoirement dans Neurone.java.
		// ----------------------------------------------------------
		Neurone neurone = creerNeurone(TYPE_NEURONE, nbEntrees);
		if (neurone == null) return;

		// ----------------------------------------------------------
		// 5. Apprentissage supervisé
		//    La méthode apprentissage() de Neurone.java ajuste les poids
		//    itérativement jusqu'à ce que la MSE passe sous MSE_LIMITE
		//    ou que MAX_ITERATIONS soit atteint.
		// ----------------------------------------------------------
		System.out.println("--- Apprentissage ---");
		neurone.apprentissage(entreesTrain, labelsTrain, MSE_LIMITE);
		System.out.println("Apprentissage terminé.\n");

		// ----------------------------------------------------------
		// 6. Évaluation sur les données d'entraînement
		//    Vérification de cohérence : le neurone doit bien classer
		//    les images qu'il a déjà vues. Si ce résultat est mauvais,
		//    l'apprentissage n'a pas fonctionné.
		// ----------------------------------------------------------
		System.out.println("--- Résultats sur TRAIN ---");
		evaluer(neurone, entreesTrain, labelsTrain, "Train");

		// ----------------------------------------------------------
		// 7. Évaluation sur les données de test
		//    Ces images n'ont jamais été vues pendant l'apprentissage.
		//    Ce résultat mesure la vraie capacité de généralisation
		//    du neurone sur des données nouvelles.
		// ----------------------------------------------------------
		System.out.println("\n--- Chargement des données de test ---");
		List<Image> imagesTest = chargerImages(DOSSIER_TEST);
		if (imagesTest == null || imagesTest.isEmpty()) {
			System.err.println("Aucune image trouvée dans " + DOSSIER_TEST);
			return;
		}
		System.out.println(imagesTest.size() + " images de test chargées.\n");

		// Même conversion/normalisation que pour les données d'entraînement
		float[][] entreesTest = new float[imagesTest.size()][nbEntrees];
		float[]   labelsTest  = new float[imagesTest.size()];
		for (int i = 0; i < imagesTest.size(); ++i) {
			Image img = imagesTest.get(i);
			int[] donnees = img.donnees();
			for (int j = 0; j < nbEntrees; ++j)
				entreesTest[i][j] = donnees[j] / 255.0f;
			labelsTest[i] = (img.label() == Image.LabelChat) ? 1.0f : 0.0f;
		}

		System.out.println("--- Résultats sur TEST ---");
		evaluer(neurone, entreesTest, labelsTest, "Test");
	}

	/**
	 * Charge toutes les images d'un dossier, les labellise selon le nom du sous-dossier
	 * (cat -> LabelChat, dog -> LabelChien, wild -> LabelWild),
	 * et les redimensionne à IMAGE_SIZE x IMAGE_SIZE par sous-échantillonnage.
	 *
	 * @param dossier chemin du dossier racine à parcourir récursivement
	 * @return liste des images chargées et redimensionnées
	 */
	static List<Image> chargerImages(String dossier)
	{
		// Récupération récursive de tous les chemins de fichiers du dossier
		List<String> chemins = Image.listeFichiers(dossier);
		if (chemins == null) return null;

		List<Image> images = new ArrayList<>();
		int erreurs = 0;
		for (String chemin : chemins) {
			// Ignorer les fichiers non-images (.gitignore, .DS_Store, etc.)
			String lower = chemin.toLowerCase();
			if (!lower.endsWith(".jpg") && !lower.endsWith(".jpeg") && !lower.endsWith(".png"))
				continue;

			// Labellisation automatique selon le nom du sous-dossier dans le chemin
			int label;
			if      (chemin.contains("cat"))  label = Image.LabelChat;
			else if (chemin.contains("dog"))  label = Image.LabelChien;
			else if (chemin.contains("wild")) label = Image.LabelWild;
			else                              label = Image.LabelInconnu;

			// Lecture de l'image depuis le disque (conversion en niveaux de gris si demandé)
			Image img = new Image(chemin, label, NIVEAUX_DE_GRIS);
			if (img.donnees() == null) { erreurs++; continue; } // image illisible

			// Redimensionnement à IMAGE_SIZE x IMAGE_SIZE par sous-échantillonnage
			Image imgReduite = souséchantillonner(img);
			if (imgReduite != null) images.add(imgReduite);
			else erreurs++;
		}
		if (erreurs > 0)
			System.out.println("  (⚠ " + erreurs + " images ignorées)");
		return images;
	}

	/**
	 * Redimensionne une image source à IMAGE_SIZE x IMAGE_SIZE par sous-échantillonnage
	 * (méthode du plus proche voisin : on sélectionne 1 pixel représentatif par bloc).
	 * Simple et rapide, mais moins précis qu'une interpolation bilinéaire.
	 *
	 * @param src image source (taille quelconque)
	 * @return nouvelle image de taille IMAGE_SIZE x IMAGE_SIZE
	 */
	static Image souséchantillonner(Image src)
	{
		if (src.donnees() == null) return null;
		int w = src.largeur();
		int h = src.hauteur();
		int canaux = NIVEAUX_DE_GRIS ? 1 : 3; // 1 canal en gris, 3 canaux en RGB
		int[] nouvellesDonnees = new int[IMAGE_SIZE * IMAGE_SIZE * canaux];

		for (int i = 0; i < IMAGE_SIZE; ++i) {
			for (int j = 0; j < IMAGE_SIZE; ++j) {
				// Calcul du pixel source correspondant (mise à l'échelle proportionnelle)
				int si = (int)(i * (float)h / IMAGE_SIZE); // ligne source
				int sj = (int)(j * (float)w / IMAGE_SIZE); // colonne source
				int indexSrc = si * w + sj;                // index dans le tableau source (ligne*largeur+colonne)
				int indexDst = i * IMAGE_SIZE + j;         // index dans le tableau destination

				if (NIVEAUX_DE_GRIS) {
					nouvellesDonnees[indexDst] = src.donnees()[indexSrc];
				} else {
					// En RGB, chaque pixel occupe 3 cases consécutives (R, G, B)
					nouvellesDonnees[3*indexDst+0] = src.donnees()[3*indexSrc+0]; // rouge
					nouvellesDonnees[3*indexDst+1] = src.donnees()[3*indexSrc+1]; // vert
					nouvellesDonnees[3*indexDst+2] = src.donnees()[3*indexSrc+2]; // bleu
				}
			}
		}
		// Construction d'un nouvel objet Image avec les données redimensionnées
		return new Image(src.label(), IMAGE_SIZE, IMAGE_SIZE, nouvellesDonnees);
	}

	/**
	 * Mélange aléatoire synchronisé des entrées et labels (algorithme de Fisher-Yates).
	 * On parcourt de la fin vers le début et on échange chaque élément avec un élément
	 * aléatoire situé avant lui. Garantit une distribution uniforme des permutations.
	 * L'échange des labels est synchronisé avec celui des entrées pour conserver
	 * l'association entrees[i] <-> labels[i].
	 */
	static void melanger(float[][] entrees, float[] labels)
	{
		Random rng = new Random();
		for (int i = entrees.length - 1; i > 0; --i) {
			int j = rng.nextInt(i + 1); // j aléatoire dans [0, i]
			// Échange entrees[i] <-> entrees[j]
			float[] tmpE = entrees[i]; entrees[i] = entrees[j]; entrees[j] = tmpE;
			// Échange labels[i] <-> labels[j] (synchronisé)
			float   tmpL = labels[i];  labels[i]  = labels[j];  labels[j]  = tmpL;
		}
	}

	/**
	 * Instancie le neurone correspondant au TYPE_NEURONE défini en CONFIGURATION.
	 * Chaque type hérite de Neurone et ne diffère que par sa fonction d'activation.
	 *
	 * @param type      "heavyside", "sigmoide" ou "relu"
	 * @param nbEntrees nombre d'entrées du neurone (= nombre de pixels de l'image)
	 * @return instance du neurone, ou null si le type est inconnu
	 */
	static Neurone creerNeurone(String type, int nbEntrees)
	{
		switch (type.toLowerCase()) {
			case "heavyside": return new NeuroneHeavyside(nbEntrees);
			case "sigmoide":  return new NeuroneSigmoide(nbEntrees);
			case "relu":      return new NeuroneReLu(nbEntrees);
			default:
				System.err.println("Type de neurone inconnu : " + type);
				return null;
		}
	}

	/**
	 * Évalue la précision du neurone sur un jeu de données et affiche :
	 * - le taux de précision global (nb corrections / nb total)
	 * - la matrice de confusion : vrais/faux positifs et négatifs
	 *
	 * Seuil de décision fixé à 0.5 : sortie >= 0.5 => chat (1), sinon chien (0).
	 * Ce seuil est valable pour Sigmoïde et ReLU ; avec Heaviside la sortie est
	 * déjà binaire (0 ou 1) donc le seuil n'a pas d'effet.
	 *
	 * Rappel : 50% de précision = équivalent à un tirage pile/face aléatoire.
	 *
	 * @param neurone le neurone entraîné à évaluer
	 * @param entrees tableau des pixels normalisés des images à tester
	 * @param labels  tableau des labels attendus (1=chat, 0=chien)
	 * @param nomJeu  nom du jeu de données affiché ("Train" ou "Test")
	 */
	static void evaluer(Neurone neurone, float[][] entrees, float[] labels, String nomJeu)
	{
		int corrects = 0;
		int vraisPositifs = 0, fauxPositifs = 0;
		int vraisNegatifs = 0, fauxNegatifs = 0;

		for (int i = 0; i < entrees.length; ++i) {
			neurone.metAJour(entrees[i]);             // calcule la sortie du neurone
			int prediction = neurone.sortie() >= 0.5f ? 1 : 0; // seuil de décision
			int verite      = (int) labels[i];

			if (prediction == verite) corrects++;

			// Matrice de confusion (positif = chat, négatif = chien)
			if      (prediction == 1 && verite == 1) vraisPositifs++; // chat prédit comme chat ✓
			else if (prediction == 1 && verite == 0) fauxPositifs++;  // chien prédit comme chat ✗
			else if (prediction == 0 && verite == 0) vraisNegatifs++; // chien prédit comme chien ✓
			else                                     fauxNegatifs++;  // chat prédit comme chien ✗
		}

		double precision = 100.0 * corrects / entrees.length;
		System.out.printf("  Précision (%s) : %d / %d = %.2f%%\n",
			nomJeu, corrects, entrees.length, precision);
		System.out.printf("  Vrais+: %d  Faux+: %d  Vrais-: %d  Faux-: %d\n",
			vraisPositifs, fauxPositifs, vraisNegatifs, fauxNegatifs);

		// Interprétation du résultat
		if      (precision < 40.0) System.out.println("  Précision inférieure au hasard — problème dans la chaîne.");
		else if (precision < 60.0) System.out.println("  Précision faible, proche du hasard (50%).");
		else if (precision < 80.0) System.out.println("  Précision correcte, peut être améliorée.");
		else if (precision < 95.0) System.out.println("  Précision satisfaisante.");
		else                       System.out.println("  Précision supérieure à 95% !");
	}
}

/*

Compiler avec :
	javac neurone/*.java Image.java CompilationAnimal.java
	java -cp ".;neurone" CompilationAnimal

*/