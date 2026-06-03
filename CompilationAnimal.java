import java.util.*;

/**
 * Programme principal de classification chats / chiens / wild.
 *
 * Au démarrage, l'utilisateur répond à 5 questions :
 *   1. Mode de classification: 3 neurones (chat+chien+wild) / 1 neurone (chat vs reste)
 *   2. Mode couleur          : tsl / rgb / gris
 *   3. Traitement de l'image : toutes / egalisation histogramme / miroir / aucun
 *   4. Ordre d'apprentissage : aleatoire / lineaire
 *   5. Type(s) de neurone    : all / sigmoide / relu / heavyside
 *
 * Architecture :
 *   - En mode 1 neurone  : un seul neurone, actif (≥0.5) = chat, inactif = pas chat
 *   - En mode 3 neurones : apprentissage compétitif, les 3 neurones apprennent
 *                          ensemble à chaque itération (plus rapide que 3x séparé)
 *
 * Compiler et lancer :
 *   javac -d class neurone/*.java Image.java CompilationAnimal.java
 *   java -cp class CompilationAnimal
 */
public class CompilationAnimal
{
	// =========================================================
	// CONSTANTES FIXES
	// =========================================================

	/** Dossier contenant les images d'entraînement (sous-dossiers cat/ dog/ wild/) */
	static final String DOSSIER_TRAIN = "dataset_groupe_6/train/";

	/** Dossier contenant les images de test (sous-dossiers cat/ dog/ wild/) */
	static final String DOSSIER_TEST  = "dataset_groupe_6/test/";

	/**
	 * MSE (Mean Squared Error) cible pour arrêter l'apprentissage.
	 * L'apprentissage s'arrête dès que l'erreur moyenne passe sous cette valeur.
	 */
	static final float MSE_LIMITE = 0.01f;

	/**
	 * Taille de redimensionnement des images en pixels (IMAGE_SIZE x IMAGE_SIZE).
	 * Ex : 64 → chaque image devient 64×64 = 4096 pixels (ou 3×4096 en RGB/TSL).
	 */
	static final int IMAGE_SIZE = 64;

	/**
	 * Nombre maximum d'itérations (epochs) pour éviter une boucle infinie.
	 * Une itération = un passage complet sur toutes les images d'entraînement.
	 */
	static final int MAX_ITERATIONS = 100;

	/**
	 * Taux d'apprentissage : contrôle l'amplitude de la correction des poids
	 * à chaque image. Trop grand = instabilité, trop petit = convergence lente.
	 */
	static final float ETA = 0.002f;

	/** Noms des 3 classes, utilisés pour l'affichage des résultats */
	static final String[] NOMS_CLASSES = {"Chat", "Chien", "Wild"};

	// =========================================================
	// DONNÉES COLLECTÉES PENDANT L'EXPÉRIENCE (pour affichage final)
	// =========================================================

	/**
	 * Historique MSE par itération pour le neurone en cours d'entraînement.
	 * Réinitialisé à chaque appel de lancerExperience().
	 * Utilisé pour tracer la courbe ASCII en fin d'expérience.
	 */
	static List<Double> historiqueMSE = new ArrayList<>();

	/**
	 * Historique des temps par itération (en ms).
	 * Réinitialisé à chaque appel de lancerExperience().
	 */
	static List<Long> historiqueTemps = new ArrayList<>();

	/**
	 * Lignes du rapport final accumulées au fur et à mesure des expériences.
	 * Chaque expérience ajoute une entrée résumant ses résultats.
	 * Affiché d'un bloc après toutes les expériences.
	 */
	static List<String> lignesRapport = new ArrayList<>();

	// =========================================================
	// CONFIGURATION CHOISIE AU DÉMARRAGE (remplie par le menu)
	// =========================================================

	/** Types de neurones à tester, ex: {"sigmoide"} ou {"heavyside","relu","sigmoide"} */
	static String[] typesALancer;

	/** true = index tiré aléatoirement à chaque pas, false = ordre du tableau */
	static boolean ordreAleatoire;

	/** true = 3 neurones (chat/chien/wild), false = 1 neurone (chat vs reste) */
	static boolean troisNeurones;

	/** Mode couleur : "gris" (1 canal), "rgb" (3 canaux) ou "tsl" (3 canaux HSL) */
	static String modeImage;

	/**
	 * Traitement appliqué aux images : "aucun", "miroir", "egalisation" ou "toutes".
	 * "toutes" applique successivement : image originale + miroir + egalisation
	 * (triple le nombre d'images d'entraînement pour augmenter la diversité).
	 */
	static String traitementImg;

	// =========================================================
	// POINT D'ENTRÉE
	// =========================================================

	public static void main(String[] args)
	{
		Scanner sc = new Scanner(System.in);

		System.out.println("=== Classification chats / chiens / wild ===\n");

		// ----------------------------------------------------------
		// Question 1 : mode de classification (du plus au moins complexe)
		//
		// 3 neurones : apprentissage compétitif, les 3 neurones apprennent
		//              ensemble. Plus discriminant, permet de distinguer
		//              chat / chien / wild séparément.
		// 1 neurone  : sortie ≥ 0.5 = chat, < 0.5 = pas chat.
		//              Simple, correspond aux grandes lignes du sujet.
		// ----------------------------------------------------------
		System.out.println("[ 1/5 ] Mode de classification :");
		System.out.println("  1 - 3 neurones : un par classe (chat / chien / wild)  [complexe]");
		System.out.println("  2 - 1 neurone  : chat (actif=1) vs tout le reste (inactif=0)  [simple]");
		troisNeurones = (lireEntier(sc, 1, 2) == 1);

		// ----------------------------------------------------------
		// Question 2 : espace colorimétrique (du plus au moins complexe)
		//
		// TSL  : 3 valeurs par pixel (teinte, saturation, luminosité).
		//        Représentation plus proche de la perception humaine ;
		//        la teinte est robuste aux variations d'éclairage.
		// RGB  : 3 valeurs par pixel (rouge, vert, bleu), 3×IMAGE_SIZE² entrées.
		//        Conserve l'information couleur complète.
		// Gris : 1 valeur par pixel (luminosité), IMAGE_SIZE² entrées.
		//        Rapide, ignore l'information de couleur.
		// ----------------------------------------------------------
		System.out.println("\n[ 2/5 ] Mode couleur :");
		System.out.println("  1 - TSL  (3 valeurs/pixel, teinte/saturation/luminosité)  [complexe]");
		System.out.println("  2 - RGB  (3 valeurs/pixel, " + 3*IMAGE_SIZE*IMAGE_SIZE + " entrées)");
		System.out.println("  3 - Niveaux de gris (1 valeur/pixel, " + IMAGE_SIZE*IMAGE_SIZE + " entrées)  [simple]");
		switch (lireEntier(sc, 1, 3)) {
			case 1:  modeImage = "tsl";  break;
			case 2:  modeImage = "rgb";  break;
			default: modeImage = "gris"; break;
		}

		// ----------------------------------------------------------
		// Question 3 : traitement appliqué aux images (du plus au moins complexe)
		//
		// Toutes       : applique les 3 versions de chaque image (originale +
		//                miroir + égalisation), ce qui triple le jeu d'entraînement.
		//                Meilleure généralisation, apprentissage plus long.
		// Egalisation  : redistribue les niveaux de pixel sur [0-255] pour maximiser
		//                le contraste. Aide le neurone sur les images ternes ou surexposées.
		// Miroir       : chaque image est inversée horizontalement.
		//                Augmente la variété des données sans collecter de nouvelles images.
		// Aucun        : images brutes redimensionnées.
		// ----------------------------------------------------------
		System.out.println("\n[ 3/5 ] Traitement de l'image :");
		System.out.println("  1 - Toutes (originale + miroir + egalisation)  [complexe, 3x plus de données]");
		System.out.println("  2 - Egalisation d'histogramme (améliore le contraste)");
		System.out.println("  3 - Miroir horizontal");
		System.out.println("  4 - Aucun  [simple]");
		switch (lireEntier(sc, 1, 4)) {
			case 1:  traitementImg = "toutes";      break;
			case 2:  traitementImg = "egalisation"; break;
			case 3:  traitementImg = "miroir";      break;
			default: traitementImg = "aucun";       break;
		}

		// ----------------------------------------------------------
		// Question 4 : ordre de présentation des images (du plus au moins complexe)
		//
		// Aléatoire : index tiré au sort à chaque pas → évite les biais d'ordre,
		//             comportement original de Neurone.java.
		// Linéaire  : on parcourt le tableau dans l'ordre → reproductible mais
		//             peut créer des biais si les données sont mal mélangées.
		// ----------------------------------------------------------
		System.out.println("\n[ 4/5 ] Ordre d'apprentissage :");
		System.out.println("  1 - Aléatoire (index tiré au sort à chaque itération)  [complexe]");
		System.out.println("  2 - Linéaire (images dans l'ordre du tableau)  [simple]");
		ordreAleatoire = (lireEntier(sc, 1, 2) == 1);

		// ----------------------------------------------------------
		// Question 5 : type(s) de neurone (du plus au moins complexe)
		// ----------------------------------------------------------
		System.out.println("\n[ 5/5 ] Type de neurone :");
		System.out.println("  1 - Tous (Heaviside + ReLU + Sigmoide)  [complexe]");
		System.out.println("  2 - Sigmoide uniquement");
		System.out.println("  3 - ReLU uniquement");
		System.out.println("  4 - Heaviside uniquement  [simple]");
		switch (lireEntier(sc, 1, 4)) {
			case 1:  typesALancer = new String[]{"heavyside", "relu", "sigmoide"}; break;
			case 2:  typesALancer = new String[]{"sigmoide"}; break;
			case 3:  typesALancer = new String[]{"relu"};     break;
			default: typesALancer = new String[]{"heavyside"}; break;
		}

		// ----------------------------------------------------------
		// Récapitulatif
		// ----------------------------------------------------------
		System.out.println("\n=== Configuration ===");
		System.out.println("  Classification: " + (troisNeurones ? "3 neurones (chat/chien/wild)" : "1 neurone (chat vs reste)"));
		System.out.println("  Mode image    : " + modeImage);
		System.out.println("  Traitement    : " + traitementImg);
		System.out.println("  Ordre         : " + (ordreAleatoire ? "aléatoire" : "linéaire"));
		System.out.println("  Neurones      : " + Arrays.toString(typesALancer));
		System.out.println("  Itérations max: " + MAX_ITERATIONS);
		System.out.println("  MSE limite    : " + MSE_LIMITE);
		System.out.println("====================\n");

		// ----------------------------------------------------------
		// Chargement des images (fait une seule fois pour tous les types de neurones)
		// ----------------------------------------------------------
		System.out.println("--- Chargement des données d'entraînement ---");
		List<Image> imagesTrain = chargerImages(DOSSIER_TRAIN);
		if (imagesTrain == null || imagesTrain.isEmpty()) {
			System.err.println("Aucune image trouvée dans " + DOSSIER_TRAIN); return;
		}
		System.out.println(imagesTrain.size() + " images chargées.\n");

		System.out.println("--- Chargement des données de test ---");
		List<Image> imagesTest = chargerImages(DOSSIER_TEST);
		if (imagesTest == null || imagesTest.isEmpty()) {
			System.err.println("Aucune image trouvée dans " + DOSSIER_TEST); return;
		}
		System.out.println(imagesTest.size() + " images de test chargées.\n");

		// ----------------------------------------------------------
		// Conversion en tableaux float[][] pour le neurone
		//
		// entrees[i] = pixels normalisés [0.0-1.0] de l'image i
		// labels[i]  = tableau de 1 ou 3 valeurs (encodage one-hot) :
		//   mode 1 neurone  : [1.0] si chat, [0.0] sinon
		//   mode 3 neurones : [1,0,0] chat / [0,1,0] chien / [0,0,1] wild
		// ----------------------------------------------------------
		final int nbEntrees = imagesTrain.get(0).taille();
		final int nbClasses = troisNeurones ? 3 : 1;

		// En mode "toutes", chaque image génère 3 versions (originale + miroir + égalisation).
		// Le test reste non-augmenté : on évalue toujours sur des images réelles.
		final int facteurAugmentation = traitementImg.equals("toutes") ? 3 : 1;

		float[][] entreesTrain = new float[imagesTrain.size() * facteurAugmentation][nbEntrees];
		float[][] labelsTrain  = new float[imagesTrain.size() * facteurAugmentation][nbClasses];
		remplirTableaux(imagesTrain, entreesTrain, labelsTrain);

		// Le jeu de test n'est jamais augmenté (évaluation sur images réelles uniquement)
		float[][] entreesTest = new float[imagesTest.size()][nbEntrees];
		float[][] labelsTest  = new float[imagesTest.size()][nbClasses];
		String traitementSauvegarde = traitementImg;
		traitementImg = "aucun"; // force : pas d'augmentation sur le test
		remplirTableaux(imagesTest, entreesTest, labelsTest);
		traitementImg = traitementSauvegarde;

		// Mélange aléatoire des données d'entraînement (Fisher-Yates)
		// Garantit que le neurone ne voit pas tous les chats d'abord, puis tous les chiens.
		melangerMultiLabels(entreesTrain, labelsTrain);
		System.out.println("Données mélangées.\n");

		// ----------------------------------------------------------
		// Lancement d'une expérience par type de neurone sélectionné
		// Les images ne sont chargées qu'une fois ; chaque expérience
		// repart de poids aléatoires frais (nouveau creerNeurone()).
		// ----------------------------------------------------------
		for (String type : typesALancer)
			lancerExperience(type, nbEntrees, nbClasses,
				entreesTrain, labelsTrain, entreesTest, labelsTest);

		// ── Rapport de synthèse final (toutes expériences) ─────────
		afficherRapportFinal();
	}

	// =========================================================
	// LECTURE CONSOLE
	// =========================================================

	/**
	 * Lit un entier dans [min, max] depuis la console.
	 * Redemande tant que la saisie est invalide ou hors plage.
	 */
	static int lireEntier(Scanner sc, int min, int max)
	{
		int valeur = min - 1;
		while (valeur < min || valeur > max) {
			System.out.print("  Votre choix (" + min + "-" + max + ") : ");
			try { valeur = Integer.parseInt(sc.nextLine().trim()); }
			catch (NumberFormatException e) { valeur = min - 1; }
			if (valeur < min || valeur > max)
				System.out.println("  Choix invalide.");
		}
		return valeur;
	}

	// =========================================================
	// CONVERSION IMAGES → TABLEAUX FLOAT
	// =========================================================

	/**
	 * Remplit les tableaux entrees[][] et labels[][] à partir d'une liste d'images.
	 *
	 * En mode "toutes", chaque image est insérée trois fois :
	 *   - telle quelle (originale)
	 *   - après miroir horizontal
	 *   - après égalisation d'histogramme
	 * Cela triple le jeu d'entraînement sans collecter de nouvelles données.
	 *
	 * En mode "miroir" ou "egalisation", un seul traitement est appliqué.
	 * En mode "aucun", les pixels bruts sont utilisés.
	 *
	 * La normalisation divise chaque pixel par 255 pour ramener les valeurs
	 * de [0-255] à [0.0-1.0], ce qui stabilise et accélère l'apprentissage.
	 *
	 * @param images  liste source d'objets Image
	 * @param entrees tableau de sortie : pixels normalisés
	 * @param labels  tableau de sortie : labels one-hot (1 ou 3 colonnes)
	 */
	static void remplirTableaux(List<Image> images, float[][] entrees, float[][] labels)
	{
		final int nbEntrees = entrees[0].length;
		final int nbClasses = labels[0].length;

		// Indice courant dans les tableaux de sortie
		int idx = 0;

		for (Image img : images) {
			// Prépare la liste des versions de données à insérer
			List<int[]> versions = new ArrayList<>();

			int[] original = img.donnees();

			if (traitementImg.equals("toutes")) {
				// Version 1 : originale
				versions.add(original);
				// Version 2 : miroir horizontal
				versions.add(appliquerMiroir(original, img.largeur(), img.hauteur()));
				// Version 3 : égalisation d'histogramme
				versions.add(egalisationHistogramme(original));
			} else {
				switch (traitementImg) {
					case "miroir":      versions.add(appliquerMiroir(original, img.largeur(), img.hauteur())); break;
					case "egalisation": versions.add(egalisationHistogramme(original));                        break;
					default:            versions.add(original); break; // "aucun"
				}
			}

			// Normalisation [0-255] → [0.0-1.0] et remplissage des tableaux
			for (int[] donnees : versions) {
				if (idx >= entrees.length) break; // sécurité : ne pas dépasser la taille allouée

				for (int j = 0; j < nbEntrees; j++)
					entrees[idx][j] = donnees[j] / 255.0f;

				// Encodage one-hot des labels
				if (nbClasses == 3) {
					labels[idx][0] = (img.label() == Image.LabelChat)  ? 1.0f : 0.0f;
					labels[idx][1] = (img.label() == Image.LabelChien) ? 1.0f : 0.0f;
					labels[idx][2] = (img.label() == Image.LabelWild)  ? 1.0f : 0.0f;
				} else {
					// Mode 1 neurone : chat = 1, tout le reste = 0
					labels[idx][0] = (img.label() == Image.LabelChat) ? 1.0f : 0.0f;
				}
				idx++;
			}
		}
	}

	// =========================================================
	// TRAITEMENTS D'IMAGE
	// =========================================================

	/**
	 * Miroir horizontal : inverse chaque ligne de pixels gauche/droite.
	 * Pour un pixel en colonne j d'une image de largeur w,
	 * sa position miroir est la colonne (w - 1 - j).
	 *
	 * Intérêt : un chat regardant à droite devient un chat regardant à gauche,
	 * ce qui double la variété des exemples sans collecter de nouvelles images.
	 *
	 * @param donnees pixels source (1 ou 3 valeurs par pixel selon le mode couleur)
	 * @param w       largeur de l'image en pixels
	 * @param h       hauteur de l'image en pixels
	 * @return nouveau tableau avec les colonnes inversées
	 */
	static int[] appliquerMiroir(int[] donnees, int w, int h)
	{
		// 1 canal pour gris, 3 canaux pour RGB/TSL
		final int canaux = donnees.length / (w * h);
		int[] miroir = new int[donnees.length];

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				int src = (i * w + j)           * canaux;
				int dst = (i * w + (w - 1 - j)) * canaux;
				for (int c = 0; c < canaux; c++)
					miroir[dst + c] = donnees[src + c];
			}
		}
		return miroir;
	}

	/**
	 * Égalisation d'histogramme : améliore le contraste d'une image.
	 *
	 * Principe en 3 étapes :
	 *   1. Histogramme : on compte combien de pixels ont chaque valeur (0 à 255).
	 *   2. Distribution cumulée : pour chaque valeur v, combien de pixels ont une
	 *      valeur ≤ v ? Cela donne la "courbe de répartition" des intensités.
	 *   3. Table de correspondance (LUT) : on étire cette courbe pour qu'elle couvre
	 *      uniformément [0-255]. Chaque valeur d'origine est remplacée par sa
	 *      valeur étirée.
	 *
	 * Résultat : une image terne (pixels concentrés entre 100 et 180) devient
	 * contrastée (pixels répartis sur 0-255). Les détails fins deviennent visibles.
	 *
	 * En RGB/TSL, l'égalisation est appliquée canal par canal pour ne pas
	 * dénaturer les couleurs.
	 *
	 * CORRECTION : cdfMin est maintenant la première valeur non nulle de
	 * l'histogramme brut, et non du cumulé, conformément à la formule standard.
	 *
	 * @param donnees pixels source
	 * @return nouveau tableau avec le contraste amélioré
	 */
	static int[] egalisationHistogramme(int[] donnees)
	{
		final int canaux   = modeImage.equals("gris") ? 1 : 3;
		final int nbPixels = donnees.length / canaux;
		int[] result = new int[donnees.length];

		for (int c = 0; c < canaux; c++) {
			// Étape 1 : histogramme du canal c
			int[] histo = new int[256];
			for (int p = 0; p < nbPixels; p++)
				histo[donnees[p * canaux + c]]++;

			// Étape 2 : trouver cdfMin = fréquence de la première valeur non nulle
			// (correction du bug original qui utilisait le cumulé à la place)
			int cdfMin = 0;
			for (int v = 0; v < 256; v++) {
				if (histo[v] > 0) { cdfMin = histo[v]; break; }
			}

			// Étape 3 : construction de la LUT via distribution cumulée
			int[] lut   = new int[256];
			int   cumul = 0;
			for (int v = 0; v < 256; v++) {
				cumul += histo[v];
				// Formule standard d'égalisation (normalisation de la CDF)
				lut[v] = (nbPixels > cdfMin)
					? Math.max(0, Math.min(255, (int)(((float)(cumul - cdfMin) / (nbPixels - cdfMin)) * 255)))
					: 0;
			}

			// Étape 4 : application de la LUT à chaque pixel du canal
			for (int p = 0; p < nbPixels; p++)
				result[p * canaux + c] = lut[donnees[p * canaux + c]];
		}
		return result;
	}

	// =========================================================
	// EXPÉRIENCE
	// =========================================================

	/**
	 * Lance une expérience complète pour un type de neurone donné :
	 *   1. Crée les neurones (poids initialisés aléatoirement)
	 *   2. Entraîne via apprentissage compétitif (tous les neurones ensemble)
	 *   3. Évalue sur TRAIN puis sur TEST
	 *
	 * @param type         "heavyside", "relu" ou "sigmoide"
	 * @param nbEntrees    nombre de pixels par image (= nb d'entrées du neurone)
	 * @param nbClasses    1 ou 3
	 * @param entreesTrain pixels normalisés des images d'entraînement
	 * @param labelsTrain  labels one-hot des images d'entraînement
	 * @param entreesTest  pixels normalisés des images de test
	 * @param labelsTest   labels one-hot des images de test
	 */
	static void lancerExperience(String type, int nbEntrees, int nbClasses,
		float[][] entreesTrain, float[][] labelsTrain,
		float[][] entreesTest,  float[][] labelsTest)
	{
		// Réinitialisation des historiques pour cette expérience
		historiqueMSE    = new ArrayList<>();
		historiqueTemps  = new ArrayList<>();

		System.out.println("+-------------------------------------------+");
		System.out.printf ("|  Experience : %-27s |\n", type);
		System.out.println("+-------------------------------------------+\n");

		// Création des neurones avec poids aléatoires frais
		Neurone[] neurones = new Neurone[nbClasses];
		for (int c = 0; c < nbClasses; c++) {
			neurones[c] = creerNeurone(type, nbEntrees);
			if (neurones[c] == null) return;
		}

		System.out.println("--- Apprentissage (détail par itération) ---");
		long debutTotal = System.currentTimeMillis();
		apprendreCompetitif(neurones, entreesTrain, labelsTrain);
		long dureeTotal = System.currentTimeMillis() - debutTotal;
		System.out.println();

		// ── Résumé compact ──────────────────────────────────────────
		afficherResumé(type, neurones, entreesTrain, labelsTrain,
		               entreesTest, labelsTest, nbClasses, dureeTotal);
	}

	/**
	 * Boucle d'apprentissage compétitif : tous les neurones apprennent ensemble.
	 *
	 * À chaque image, les N neurones calculent leur sortie simultanément,
	 * puis chacun corrige ses poids selon l'écart entre sa sortie et sa cible
	 * (1.0 pour la bonne classe, 0.0 pour les autres).
	 *
	 * Avantages vs apprentissage séquentiel (neurone par neurone) :
	 *   - 3× plus rapide (500 itérations au lieu de 1500)
	 *   - Les neurones apprennent en contexte : le neurone "chien" sait
	 *     qu'il concurrence "chat" et "wild", ce qui le rend plus discriminant.
	 *
	 * La MSE est calculée sur tous les neurones ensemble (moyenne globale).
	 * L'apprentissage s'arrête quand MSE < MSE_LIMITE ou iter >= MAX_ITERATIONS.
	 *
	 * ATTENTION : synapses() doit retourner le tableau interne par référence
	 * (et non une copie) pour que la mise à jour des poids soit effective.
	 * Vérifier l'implémentation de Neurone.java si les poids ne convergent pas.
	 *
	 * @param neurones tableau des neurones à entraîner (1 ou 3)
	 * @param entrees  pixels normalisés des images d'entraînement
	 * @param labels   labels one-hot (1 ou 3 colonnes selon le mode)
	 */
	static void apprendreCompetitif(Neurone[] neurones,
		float[][] entrees,
		float[][] labels)
	{
		final Random rng = new Random();
		final int n  = entrees.length;
		final int nc = neurones.length;

		double mse;
		int iter = 0;

		// ------------------------------------------------------
		// FIX #1 : allocations déplacées hors boucle chaude
		// ------------------------------------------------------

		// Réutilisé à chaque image (évite des millions de new float[])
		final float[] sorties = new float[nc];

		// Références directes des poids des neurones
		// (évite de rappeler synapses() en permanence)
		final float[][] synapsesNeurones = new float[nc][];

		for (int c = 0; c < nc; c++)
			synapsesNeurones[c] = neurones[c].synapses();

		do {
			long debutIter = System.currentTimeMillis();
			mse = 0.0;

			for (int i = 0; i < n; i++) {

				final int idx =
					ordreAleatoire ? rng.nextInt(n) : i;

				final float[] entree = entrees[idx];

				// ------------------------------
				// Forward pass
				// ------------------------------
				for (int c = 0; c < nc; c++) {
					neurones[c].metAJour(entree);
					sorties[c] = neurones[c].sortie();
				}

				// ------------------------------
				// Update poids + biais
				// ------------------------------
				for (int c = 0; c < nc; c++) {

					final float delta =
						labels[idx][c] - sorties[c];

					mse += delta * delta;

					// référence directe du tableau interne
					final float[] synapses =
						synapsesNeurones[c];

					for (int j = 0; j < entree.length; j++) {
						synapses[j] +=
							entree[j] * ETA * delta;
					}

					neurones[c].fixeBiais(
						neurones[c].biais()
						+ ETA * delta
					);
				}
			}

			mse /= (n * nc);

			long dureeIter =
				System.currentTimeMillis() - debutIter;

			historiqueMSE.add(mse);
			historiqueTemps.add(dureeIter);

			System.out.printf(
				"  Itération %3d | mse: %9.6f | %4d ms%n",
				iter,
				mse,
				dureeIter
			);

			iter++;

		} while (mse > MSE_LIMITE
			&& iter < MAX_ITERATIONS);
	}
	// =========================================================
	// RÉSUMÉ COMPACT + COURBE MSE
	// =========================================================

	/**
	 * Affiche le résumé complet d'une expérience en un bloc unique :
	 *   - Statistiques temporelles (total, moyenne, min, max par itération)
	 *   - Courbe ASCII de la MSE au fil des itérations
	 *   - Résultats Train et Test côte à côte
	 */
	static void afficherResumé(String type, Neurone[] neurones,
		float[][] entreesTrain, float[][] labelsTrain,
		float[][] entreesTest,  float[][] labelsTest,
		int nbClasses, long dureeTotal)
	{
		int nbIter = historiqueMSE.size();
		double mseFin  = historiqueMSE.get(nbIter - 1);
		double mseInit = historiqueMSE.get(0);

		// Statistiques de temps
		long tMin = Long.MAX_VALUE, tMax = 0, tSum = 0;
		for (long t : historiqueTemps) {
			if (t < tMin) tMin = t;
			if (t > tMax) tMax = t;
			tSum += t;
		}
		double tMoy = (double) tSum / nbIter;

		String ligne = "=".repeat(57);
		String tiret = "-".repeat(57);
		String bord  = "|";

		System.out.println("+" + ligne + "+");
		System.out.printf ("|  RESULTATS - %-42s|\n", type.toUpperCase());
		System.out.println("+" + tiret + "+");

		// ── Timing ─────────────────────────────────────────────────
		System.out.printf ("|  %-54s|\n", "TIMING");
		System.out.printf ("|    Iterations      : %-5d  Total : %6d ms%13s|\n",
			nbIter, dureeTotal, "");
		System.out.printf ("|    Moy/iter        : %6.1f ms  Min : %4d ms  Max : %4d ms   |\n",
			tMoy, tMin, tMax);
		System.out.println("+" + tiret + "+");

		// ── Courbe MSE ─────────────────────────────────────────────
		System.out.printf ("|  %-54s|\n", "COURBE MSE");
		tracerCourbeMSE(52);
		System.out.println("+" + tiret + "+");

		// ── MSE synthèse ───────────────────────────────────────────
		System.out.printf ("|    MSE initiale    : %-10.6f  MSE finale : %-12.6f|\n",
			mseInit, mseFin);
		String arret = (mseFin <= MSE_LIMITE) ? "Converge (MSE < limite)" : "Max iterations atteint";
		System.out.printf ("|    Arret           : %-35s|\n", arret);
		System.out.println("+" + tiret + "+");

		// ── Résultats Train ────────────────────────────────────────
		System.out.printf ("|  %-54s|\n", "PRECISION - TRAIN");
		afficherPrecisionCompacte(neurones, entreesTrain, labelsTrain, nbClasses);
		System.out.println("+" + tiret + "+");

		// ── Résultats Test ─────────────────────────────────────────
		System.out.printf ("|  %-54s|\n", "PRECISION - TEST");
		afficherPrecisionCompacte(neurones, entreesTest, labelsTest, nbClasses);
		System.out.println("+" + ligne + "+\n");

		// ── Collecte pour le rapport final ─────────────────────────
		double precTrain = calculerPrecision(neurones, entreesTrain, labelsTrain, nbClasses);
		double precTest  = calculerPrecision(neurones, entreesTest,  labelsTest,  nbClasses);
		String arretCourt = (mseFin <= MSE_LIMITE) ? "converge" : "max_iter";
		lignesRapport.add(String.format("%-12s | %3d iter | %6d ms | train:%6.2f%% | test:%6.2f%% | %s",
			type, nbIter, dureeTotal, precTrain, precTest, arretCourt));
	}

	/**
	 * Affiche le tableau de synthèse final regroupant toutes les expériences.
	 * Appelé une seule fois, après la boucle sur typesALancer dans main().
	 */
	static void afficherRapportFinal()
	{
		if (lignesRapport.isEmpty()) return;

		String sep = "=".repeat(75);
		System.out.println("\n+" + sep + "+");
		System.out.printf ("|  %-73s|\n", "RAPPORT FINAL - TOUTES LES EXPERIENCES");
		System.out.println("+" + "-".repeat(75) + "+");
		System.out.printf ("|  %-12s | %-8s | %-9s | %-16s | %-15s | %-8s|\n",
			"Neurone", "Iters", "Temps", "Train", "Test", "Arret");
		System.out.println("+" + "-".repeat(75) + "+");
		for (String ligne : lignesRapport)
			System.out.printf("|  %s|\n", ligne);
		System.out.println("+" + sep + "+\n");
	}

	/**
	 * Calcule la précision globale sans afficher quoi que ce soit.
	 * Utilisé uniquement pour collecter la ligne du rapport final.
	 */
	static double calculerPrecision(Neurone[] neurones, float[][] entrees,
		float[][] labels, int nbClasses)
	{
		int corrects = 0;
		for (int i = 0; i < entrees.length; i++) {
			if (nbClasses == 1) {
				neurones[0].metAJour(entrees[i]);
				int pred   = neurones[0].sortie() >= 0.5f ? 1 : 0;
				int verite = (int) labels[i][0];
				if (pred == verite) corrects++;
			} else {
				float[] sorties = new float[nbClasses];
				for (int c = 0; c < nbClasses; c++) {
					neurones[c].metAJour(entrees[i]);
					sorties[c] = neurones[c].sortie();
				}
				if (argmax(sorties) == argmax(labels[i])) corrects++;
			}
		}
		return 100.0 * corrects / entrees.length;
	}

	/**
	 * Trace une courbe ASCII de l'historique MSE dans une largeur fixée.
	 * La hauteur est de 8 lignes, chaque point représente une ou plusieurs itérations.
	 * La courbe est encadrée pour s'insérer proprement dans le tableau de résumé.
	 *
	 * @param largeur nombre de colonnes intérieures disponibles (hors bordures ║)
	 */
	static void tracerCourbeMSE(int largeur)
	{
		final int HAUTEUR = 8;
		int nbIter = historiqueMSE.size();

		// Calcul min/max MSE pour normaliser la hauteur
		double mseMax = historiqueMSE.stream().mapToDouble(d -> d).max().orElse(1.0);
		double mseMin = historiqueMSE.stream().mapToDouble(d -> d).min().orElse(0.0);
		// Léger padding en bas pour que la ligne finale soit visible
		double plage = Math.max(mseMax - mseMin, 1e-9);

		// Echantillonnage : on réduit ou étend les points à 'largeur' colonnes
		double[] points = new double[largeur];
		for (int col = 0; col < largeur; col++) {
			// Index dans l'historique correspondant à cette colonne
			int idx = (int)((col / (double)(largeur - 1)) * (nbIter - 1));
			idx = Math.min(idx, nbIter - 1);
			points[col] = historiqueMSE.get(idx);
		}

		// Remplissage de la grille caractère par caractère
		char[][] grille = new char[HAUTEUR][largeur];
		for (char[] row : grille) Arrays.fill(row, ' ');

		for (int col = 0; col < largeur; col++) {
			// Ligne 0 = haut (mseMax), ligne HAUTEUR-1 = bas (mseMin)
			int ligne = (int)((mseMax - points[col]) / plage * (HAUTEUR - 1));
			ligne = Math.max(0, Math.min(HAUTEUR - 1, ligne));
			grille[ligne][col] = 'o';  // ASCII pur, visible sur toutes les consoles
		}

		// Affichage avec axes et étiquettes MSE sur bordure gauche
		for (int row = 0; row < HAUTEUR; row++) {
			double valAxe = mseMax - row * (plage / (HAUTEUR - 1));
			String label = (row % 2 == 0) ? String.format("%7.4f", valAxe) : "       ";
			System.out.printf("|  %s |%s|\n", label, new String(grille[row]));
		}
		// Axe horizontal
		System.out.printf("|         +%s+\n", "-".repeat(largeur));
		System.out.printf("|         %3s%s%3s\n",
			"0", " ".repeat(largeur - 6), String.valueOf(nbIter - 1));
		System.out.printf("|         %s iterations\n", " ".repeat(largeur / 2 - 5));
	}

	/**
	 * Calcule et affiche la précision dans le format compact du résumé.
	 * Mode 1 neurone  : précision globale + matrice de confusion sur une ligne.
	 * Mode 3 neurones : précision globale + ligne par classe.
	 */
	static void afficherPrecisionCompacte(Neurone[] neurones, float[][] entrees,
		float[][] labels, int nbClasses)
	{
		int corrects = 0;

		if (nbClasses == 1) {
			int vp = 0, fp = 0, vn = 0, fn = 0;
			for (int i = 0; i < entrees.length; i++) {
				neurones[0].metAJour(entrees[i]);
				int pred   = neurones[0].sortie() >= 0.5f ? 1 : 0;
				int verite = (int) labels[i][0];
				if (pred == verite) corrects++;
				if      (pred == 1 && verite == 1) vp++;
				else if (pred == 1 && verite == 0) fp++;
				else if (pred == 0 && verite == 0) vn++;
				else                               fn++;
			}
			double prec = 100.0 * corrects / entrees.length;
			System.out.printf("|    Global : %d/%d = %.2f%%  %s\n",
				corrects, entrees.length, prec, interpreterCourt(prec));
			System.out.printf("|    VP:%-5d FP:%-5d VN:%-5d FN:%-5d            |\n",
				vp, fp, vn, fn);

		} else {
			int[]   corP  = new int[3];
			int[]   totP  = new int[3];
			for (int i = 0; i < entrees.length; i++) {
				float[] sorties = new float[3];
				for (int c = 0; c < 3; c++) {
					neurones[c].metAJour(entrees[i]);
					sorties[c] = neurones[c].sortie();
				}
				int pred   = argmax(sorties);
				int verite = argmax(labels[i]);
				totP[verite]++;
				if (pred == verite) { corrects++; corP[verite]++; }
			}
			double prec = 100.0 * corrects / entrees.length;
			System.out.printf("|    Global : %d/%d = %.2f%%  %s\n",
				corrects, entrees.length, prec, interpreterCourt(prec));
			for (int c = 0; c < 3; c++) {
				double pc = totP[c] > 0 ? 100.0 * corP[c] / totP[c] : 0.0;
				System.out.printf("|      %-6s: %4d / %4d = %6.2f%%%25s|\n",
					NOMS_CLASSES[c], corP[c], totP[c], pc, "");
			}
		}
	}

	/** Étiquette courte d'interprétation pour l'affichage compact. */
	static String interpreterCourt(double precision)
	{
		if      (precision < 40.0) return "/!\\ < hasard         |";
		else if (precision < 60.0) return "~   hasard            |";
		else if (precision < 80.0) return "OK  correct           |";
		else if (precision < 95.0) return "++  satisfaisant      |";
		else                       return "*** excellent (>95%)  |";
	}

	// =========================================================
	// ÉVALUATION (conservée pour compatibilité interne)
	// =========================================================

	/**
	 * Évalue la précision des neurones sur un jeu de données.
	 *
	 * Mode 1 neurone  : seuil à 0.5 → sortie ≥ 0.5 = chat, sinon pas chat.
	 *                   Affiche une matrice de confusion (vrais/faux positifs).
	 * Mode 3 neurones : argmax des sorties → la classe prédite est celle dont
	 *                   le neurone a la sortie la plus haute.
	 *                   Affiche la précision globale + par classe.
	 *
	 * Rappel : 50% = équivalent à un tirage pile/face (hasard pur sur 2 classes).
	 *          Sur 3 classes équilibrées, le hasard donne ~33%.
	 *
	 * @param neurones  tableau des neurones entraînés
	 * @param entrees   pixels normalisés des images à évaluer
	 * @param labels    labels one-hot attendus
	 * @param nomJeu    "Train" ou "Test" (pour l'affichage)
	 * @param nbClasses 1 ou 3
	 */
	static void evaluer(Neurone[] neurones, float[][] entrees, float[][] labels,
		String nomJeu, int nbClasses)
	{
		int corrects = 0;

		if (nbClasses == 1) {
			// --- Mode 1 neurone : matrice de confusion ---
			int vraisPos = 0, fauxPos = 0, vraisNeg = 0, fauxNeg = 0;

			for (int i = 0; i < entrees.length; i++) {
				neurones[0].metAJour(entrees[i]);
				int prediction = neurones[0].sortie() >= 0.5f ? 1 : 0;
				int verite     = (int) labels[i][0];

				if (prediction == verite) corrects++;
				if      (prediction == 1 && verite == 1) vraisPos++;
				else if (prediction == 1 && verite == 0) fauxPos++;
				else if (prediction == 0 && verite == 0) vraisNeg++;
				else                                     fauxNeg++;
			}

			double precision = 100.0 * corrects / entrees.length;
			System.out.printf("  Précision (%s) : %d / %d = %.2f%%\n",
				nomJeu, corrects, entrees.length, precision);
			System.out.printf("  Vrais+ (chat)  : %d   Faux+ (autre prédit chat) : %d\n", vraisPos, fauxPos);
			System.out.printf("  Vrais- (autre) : %d   Faux- (chat prédit autre) : %d\n", vraisNeg, fauxNeg);
			afficherInterpretation(precision);

		} else {
			// --- Mode 3 neurones : précision globale + par classe ---
			int[] correctsParClasse = new int[3];
			int[] totalParClasse    = new int[3];

			for (int i = 0; i < entrees.length; i++) {
				float[] sorties = new float[3];
				for (int c = 0; c < 3; c++) {
					neurones[c].metAJour(entrees[i]);
					sorties[c] = neurones[c].sortie();
				}
				int prediction = argmax(sorties);
				int verite     = argmax(labels[i]);

				totalParClasse[verite]++;
				if (prediction == verite) {
					corrects++;
					correctsParClasse[verite]++;
				}
			}

			double precision = 100.0 * corrects / entrees.length;
			System.out.printf("  Précision globale (%s) : %d / %d = %.2f%%\n",
				nomJeu, corrects, entrees.length, precision);

			for (int c = 0; c < 3; c++) {
				double precClasse = totalParClasse[c] > 0
					? 100.0 * correctsParClasse[c] / totalParClasse[c] : 0.0;
				System.out.printf("    %-6s : %d / %d = %.2f%%\n",
					NOMS_CLASSES[c], correctsParClasse[c], totalParClasse[c], precClasse);
			}
			afficherInterpretation(precision);
		}
	}

	/**
	 * Affiche une interprétation textuelle du taux de précision obtenu.
	 * Seuils : <40% inférieur au hasard, <60% proche du hasard,
	 *          <80% correct, <95% satisfaisant, ≥95% excellent.
	 */
	static void afficherInterpretation(double precision)
	{
		if      (precision < 40.0) System.out.println("  -> Precision inferieure au hasard.");
		else if (precision < 60.0) System.out.println("  -> Precision faible, proche du hasard.");
		else if (precision < 80.0) System.out.println("  -> Precision correcte, peut etre amelioree.");
		else if (precision < 95.0) System.out.println("  -> Precision satisfaisante.");
		else                       System.out.println("  -> Precision superieure a 95% !");
	}

	// =========================================================
	// UTILITAIRES
	// =========================================================

	/**
	 * Retourne l'index de la valeur maximale dans un tableau.
	 * Utilisé pour déterminer la classe prédite (argmax des sorties)
	 * et la vraie classe (argmax du vecteur one-hot).
	 * Ex : [0.2, 0.8, 0.3] → 1 (le neurone "chien" est le plus actif)
	 */
	static int argmax(float[] v)
	{
		int idx = 0;
		for (int i = 1; i < v.length; i++)
			if (v[i] > v[idx]) idx = i;
		return idx;
	}

	/**
	 * Mélange aléatoire synchronisé des tableaux entrees et labels (Fisher-Yates).
	 * On parcourt de la fin vers le début et on échange chaque élément avec
	 * un élément aléatoire situé avant lui, ce qui garantit une distribution
	 * uniforme de toutes les permutations possibles.
	 * L'échange est synchronisé entre entrees et labels pour conserver
	 * l'association entrees[i] ↔ labels[i].
	 */
	static void melangerMultiLabels(float[][] entrees, float[][] labels)
	{
		final Random rng = new Random();
		for (int i = entrees.length - 1; i > 0; i--) {
			int j = rng.nextInt(i + 1);
			float[] tmpE = entrees[i]; entrees[i] = entrees[j]; entrees[j] = tmpE;
			float[] tmpL = labels[i];  labels[i]  = labels[j];  labels[j]  = tmpL;
		}
	}

	// =========================================================
	// CHARGEMENT DES IMAGES
	// =========================================================

	/**
	 * Charge toutes les images d'un dossier, les labellise selon le nom du
	 * sous-dossier (cat→LabelChat, dog→LabelChien, wild→LabelWild),
	 * applique la conversion colorimétrique choisie (gris/rgb/tsl),
	 * puis redimensionne à IMAGE_SIZE×IMAGE_SIZE par sous-échantillonnage.
	 *
	 * En mode "toutes", le nombre d'images retournées est 3× le nombre
	 * d'images sources (originale + miroir + égalisation).
	 * Note : pour le jeu de TEST, le mode "toutes" n'ajoute PAS de versions
	 * augmentées — le test doit rester représentatif des images réelles.
	 * Cette distinction est gérée en aval dans remplirTableaux().
	 *
	 * @param dossier chemin du dossier racine à parcourir récursivement
	 * @return liste des images chargées et redimensionnées
	 */
	static List<Image> chargerImages(String dossier)
	{
		List<String> chemins = Image.listeFichiers(dossier);
		if (chemins == null) return null;

		// Image.java prend un booléen niveauxDeGris.
		// Pour rgb et tsl on charge en couleur, la conversion TSL est faite ensuite.
		final boolean chargerEnGris = modeImage.equals("gris");

		List<Image> images  = new ArrayList<>();
		int         erreurs = 0;

		for (String chemin : chemins) {
			// Ignorer les fichiers non-images (.gitignore, .DS_Store, etc.)
			String lower = chemin.toLowerCase();
			if (!lower.endsWith(".jpg") && !lower.endsWith(".jpeg") && !lower.endsWith(".png"))
				continue;

			// Labellisation automatique selon le nom du sous-dossier
			int label;
			if      (chemin.contains("cat"))  label = Image.LabelChat;
			else if (chemin.contains("dog"))  label = Image.LabelChien;
			else if (chemin.contains("wild")) label = Image.LabelWild;
			else                              label = Image.LabelInconnu;

			Image img = new Image(chemin, label, chargerEnGris);
			if (img.donnees() == null) { erreurs++; continue; }

			// Conversion RGB → TSL si demandée
			if (modeImage.equals("tsl"))
				img = convertirEnTSL(img);

			Image imgReduite = sousEchantillonner(img);
			if (imgReduite != null) images.add(imgReduite);
			else erreurs++;
		}

		if (erreurs > 0)
			System.out.println("  (/!\\ " + erreurs + " images ignorees)");
		return images;
	}

	/**
	 * Redimensionne une image à IMAGE_SIZE×IMAGE_SIZE par sous-échantillonnage
	 * (méthode du plus proche voisin).
	 *
	 * Pour chaque pixel de destination (i, j), on calcule le pixel source
	 * correspondant par mise à l'échelle proportionnelle :
	 *   si = i × hauteur_source / IMAGE_SIZE
	 *   sj = j × largeur_source  / IMAGE_SIZE
	 *
	 * C'est la méthode la plus rapide mais elle peut créer des artefacts
	 * sur les petits détails (aliasing). Une interpolation bilinéaire serait
	 * plus précise mais plus coûteuse en calcul.
	 *
	 * Renommée de souséchantillonner → sousEchantillonner pour éviter les
	 * problèmes d'encodage avec le caractère accentué dans le nom de méthode.
	 *
	 * @param src image source (taille quelconque)
	 * @return nouvelle image de taille IMAGE_SIZE×IMAGE_SIZE
	 */
	static Image sousEchantillonner(Image src)
	{
		if (src.donnees() == null) return null;

		final int w      = src.largeur();
		final int h      = src.hauteur();
		final int canaux = modeImage.equals("gris") ? 1 : 3;
		int[] nouvellesDonnees = new int[IMAGE_SIZE * IMAGE_SIZE * canaux];

		for (int i = 0; i < IMAGE_SIZE; i++) {
			for (int j = 0; j < IMAGE_SIZE; j++) {
				int si       = (int)(i * (float)h / IMAGE_SIZE); // ligne source
				int sj       = (int)(j * (float)w / IMAGE_SIZE); // colonne source
				int indexSrc = si * w + sj;
				int indexDst = i * IMAGE_SIZE + j;

				if (canaux == 1) {
					nouvellesDonnees[indexDst] = src.donnees()[indexSrc];
				} else {
					nouvellesDonnees[3*indexDst+0] = src.donnees()[3*indexSrc+0]; // R ou T
					nouvellesDonnees[3*indexDst+1] = src.donnees()[3*indexSrc+1]; // G ou S
					nouvellesDonnees[3*indexDst+2] = src.donnees()[3*indexSrc+2]; // B ou L
				}
			}
		}
		return new Image(src.label(), IMAGE_SIZE, IMAGE_SIZE, nouvellesDonnees);
	}

	/**
	 * Convertit une image RGB en TSL (Teinte / Saturation / Luminosité).
	 *
	 * RGB décrit une couleur par ses composantes physiques (rouge, vert, bleu).
	 * TSL la décrit comme un humain la perçoit :
	 *   T (Teinte)      : angle sur la roue des couleurs, 0°-360° → normalisé [0-255]
	 *   S (Saturation)  : intensité de la couleur, 0=gris pur → normalisé [0-255]
	 *   L (Luminosité)  : clarté, 0=noir, 0.5=normal, 1=blanc → normalisé [0-255]
	 *
	 * Avantage : la teinte est robuste aux variations d'éclairage. Deux photos
	 * du même chat sous des lumières différentes auront des RGB très différents
	 * mais des teintes similaires.
	 *
	 * CORRECTION : le modulo sur float (% 6) est remplacé par une soustraction
	 * explicite pour éviter les instabilités numériques (valeurs -0.0 ou
	 * légèrement négatives dues aux arrondis).
	 *
	 * @param src image source en RGB
	 * @return nouvelle image avec données converties en TSL [0-255]
	 */
	static Image convertirEnTSL(Image src)
	{
		final int[] donnees  = src.donnees();
		final int   nbPixels = donnees.length / 3;
		int[] tsl = new int[donnees.length];

		for (int p = 0; p < nbPixels; p++) {
			float r = donnees[3*p+0] / 255.0f;
			float g = donnees[3*p+1] / 255.0f;
			float b = donnees[3*p+2] / 255.0f;

			float max   = Math.max(r, Math.max(g, b));
			float min   = Math.min(r, Math.min(g, b));
			float delta = max - min;

			// Luminosité
			float l = (max + min) / 2.0f;

			// Saturation (0 si pixel gris, delta nul)
			float s = (delta < 1e-6f) ? 0.0f
				: delta / (1.0f - Math.abs(2.0f * l - 1.0f));

			// Teinte en degrés [0-360], puis normalisée en [0-1]
			// CORRECTION : remplacement du % 6 flottant (instable) par une
			// soustraction explicite garantissant t ∈ [0.0, 1.0]
			float t = 0.0f;
			if (delta > 1e-6f) {
				if (max == r) {
					t = 60.0f * ((g - b) / delta);
					if (t < 0.0f) t += 360.0f;
					if (t >= 360.0f) t -= 360.0f;
				} else if (max == g) {
					t = 60.0f * ((b - r) / delta + 2.0f);
				} else {
					t = 60.0f * ((r - g) / delta + 4.0f);
				}
				t /= 360.0f;
			}

			// Stockage en [0-255] pour rester compatible avec le pipeline de normalisation
			tsl[3*p+0] = Math.min(255, (int)(t * 255)); // Teinte
			tsl[3*p+1] = Math.min(255, (int)(s * 255)); // Saturation
			tsl[3*p+2] = Math.min(255, (int)(l * 255)); // Luminosité
		}
		return new Image(src.label(), src.largeur(), src.hauteur(), tsl);
	}

	/**
	 * Instancie le neurone du type demandé avec nbEntrees entrées.
	 * Chaque type hérite de Neurone et ne diffère que par sa fonction d'activation :
	 *   - Heaviside : sortie binaire (0 ou 1), apprentissage par sauts.
	 *   - ReLU      : sortie = max(0, x), rapide mais peut saturer.
	 *   - Sigmoide  : sortie continue [0-1], convergence douce, la plus efficace.
	 *
	 * @param type      "heavyside", "sigmoide" ou "relu"
	 * @param nbEntrees nombre d'entrées (= nombre de pixels)
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
}