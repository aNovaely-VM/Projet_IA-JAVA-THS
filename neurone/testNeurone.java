import java.util.*;
import java.io.File;

public class testNeurone {
	public static void main(String[] args) {

		// =========================================================================
		// CONFIGURATION DES TESTS
		// =========================================================================
		boolean avecMelange = true;
		boolean avecNormalisation = true;
		double niveauBruit = 0.2; // Modifie pour tester la robustesse (ex: 0.1 pour 10%)
		boolean niveauxDeGris = true;
		double mseLimite = 0.005;

		String cheminTrain = "dataset_groupe_6/train";
		String cheminTest = "dataset_groupe_6/test";

		System.out.println("=======================================================");
		System.out.println(" EXPÉRIMENTATION MULTI-CLASSE : 3 NEURONES (One-vs-All)");
		System.out.println("=======================================================");

		// =========================================================================
		// ÉTAPE 1 : Chargement des images d'entraînement
		// =========================================================================
		System.out.println("\n=== ÉTAPE 1 : Chargement des images d'entraînement ===");
		List<Image> listeImagesTrain = new ArrayList<>();
		chargerImagesRec(new File(cheminTrain), listeImagesTrain, niveauxDeGris);

		System.out.println(listeImagesTrain.size() + " images chargées pour l'entraînement.");
		if (listeImagesTrain.isEmpty()) {
			System.out.println("Erreur : Le dossier d'entraînement est vide ou introuvable.");
			return;
		}

		if (avecMelange) {
			Collections.shuffle(listeImagesTrain);
			System.out.println("[Configuration] Données d'entraînement mélangées.");
		}

		int nbEntrees = listeImagesTrain.get(0).taille();
		double[][] entreesTrain = new double[listeImagesTrain.size()][nbEntrees];

		// Remplissage de la matrice d'entrées (commune aux 3 neurones)
		for (int i = 0; i < listeImagesTrain.size(); i++) {
			Image img = listeImagesTrain.get(i);
			entreesTrain[i] = avecNormalisation ? normaliserTableau(img.donnees()) : convertirEnDouble(img.donnees());
		}

		// =========================================================================
		// ÉTAPE 2 : Instanciation et Apprentissage des 3 Neurones
		// =========================================================================
		System.out.println("\n=== ÉTAPE 2 : Apprentissage des 3 neurones ===");
		int nbClasses = 3; // 0: Chat, 1: Chien, 2: Wild
		iNeurone[] reseauDeNeurones = new iNeurone[nbClasses];
		String[] nomsClasses = { "CHAT", "CHIEN", "WILD" };

		for (int c = 0; c < nbClasses; c++) {
			System.out.println("\n--- Entraînement du neurone spécialisé [" + nomsClasses[c] + "] ---");
			reseauDeNeurones[c] = new NeuroneSigmoide(nbEntrees);

			// Génération des consignes spécifiques pour ce neurone (1.0 si c'est sa classe,
			// sinon 0.0)
			double[] consignesSpecifiques = new double[listeImagesTrain.size()];
			for (int i = 0; i < listeImagesTrain.size(); i++) {
				consignesSpecifiques[i] = (listeImagesTrain.get(i).label() == c) ? 1.0 : 0.0;
			}

			// Entraînement du neurone actuel
			reseauDeNeurones[c].apprentissage(entreesTrain, consignesSpecifiques, mseLimite);
		}
		System.out.println("\nApprentissage du système multi-classe terminé !");

		// =========================================================================
		// ÉTAPE 3 : Chargement des images de test
		// =========================================================================
		System.out.println("\n=== ÉTAPE 3 : Chargement des images de test ===");
		List<Image> listeImagesTest = new ArrayList<>();
		chargerImagesRec(new File(cheminTest), listeImagesTest, niveauxDeGris);
		System.out.println(listeImagesTest.size() + " images chargées pour le test.");

		if (listeImagesTest.isEmpty()) {
			System.out.println("Erreur : Aucune image de test à évaluer.");
			return;
		}

		// =========================================================================
		// ÉTAPE 4 : Évaluation et Matrice de Confusion Multi-classe (3x3)
		// =========================================================================
		System.out.println("\n=== ÉTAPE 4 : Évaluation & Matrice de Confusion ===");

		// Matrice 3x3 : lignes = classes attendues, colonnes = classes prédites
		int[][] matriceConfusion = new int[nbClasses][nbClasses];
		int casAmbigusOuIndecis = 0;
		int succesGlobal = 0;

		for (Image img : listeImagesTest) {
			double[] entreesTest = avecNormalisation ? normaliserTableau(img.donnees())
					: convertirEnDouble(img.donnees());
			if (niveauBruit > 0.0) {
				entreesTest = appliquerBruit(entreesTest, niveauBruit, avecNormalisation);
			}

			int classePredite = -1;
			double maxActivation = -Double.MAX_VALUE;

			// RÈGLE DE L'ARGMAX : On cherche le neurone qui renvoie la plus forte valeur
			for (int c = 0; c < nbClasses; c++) {
				reseauDeNeurones[c].metAJour(entreesTest);
				double scoreSortie = reseauDeNeurones[c].sortie();

				if (scoreSortie > maxActivation) {
					maxActivation = scoreSortie;
					classePredite = c;
				}
			}

			int attendu = img.label(); // Label réel de l'image (0, 1 ou 2)

			// Si un neurone s'est démarqué, on enregistre la prédiction
			if (classePredite != -1) {
				matriceConfusion[attendu][classePredite]++;
				if (classePredite == attendu) {
					succesGlobal++;
				}
			} else {
				casAmbigusOuIndecis++;
			}
		}

		// =========================================================================
		// AFFICHAGE DU RAPPORT SCIENTIFIQUE
		// =========================================================================
		System.out.println("\n=======================================================");
		System.out.println("          MATRICE DE CONFUSION MULTI-CLASSE            ");
		System.out.println("=======================================================");
		System.out.println("Attendu \\ Prédit |  CHAT  |  CHIEN |  WILD  |");
		System.out.println("-------------------------------------------------------");
		System.out.printf("  CHAT           |   %3d  |   %3d  |   %3d  |\n", matriceConfusion[0][0],
				matriceConfusion[0][1], matriceConfusion[0][2]);
		System.out.printf("  CHIEN          |   %3d  |   %3d  |   %3d  |\n", matriceConfusion[1][0],
				matriceConfusion[1][1], matriceConfusion[1][2]);
		System.out.printf("  WILD           |   %3d  |   %3d  |   %3d  |\n", matriceConfusion[2][0],
				matriceConfusion[2][1], matriceConfusion[2][2]);
		System.out.println("-------------------------------------------------------");
		System.out.printf(" Images indécises ou conflictuelles : %d\n", casAmbigusOuIndecis);
		System.out.println("-------------------------------------------------------");

		double precisionGlobale = ((double) succesGlobal / listeImagesTest.size()) * 100.0;
		System.out.printf(" PRÉCISION GLOBALE DU SYSTÈME      : %.2f %%\n", precisionGlobale);
		System.out.println("=======================================================");
	}

	/**
	 * Méthode récursive pour explorer les dossiers et charger toutes les images
	 */
	private static void chargerImagesRec(File dossier, List<Image> liste, boolean niveauxDeGris) {
		if (!dossier.exists())
			return;

		File[] fichiers = dossier.listFiles();
		if (fichiers == null)
			return;

		for (File f : fichiers) {
			if (f.isDirectory()) {
				chargerImagesRec(f, liste, niveauxDeGris);
			} else if (f.isFile()) {
				String nom = f.getName().toLowerCase();
				if (nom.endsWith(".jpg") || nom.endsWith(".jpeg") || nom.endsWith(".png")) {
					int labelAuto = determinerLabel(f);
					liste.add(new Image(f.getAbsolutePath(), labelAuto, niveauxDeGris));
				}
			}
		}
	}

	private static int determinerLabel(File fichier) {
		String chemin = fichier.getAbsolutePath().toLowerCase();
		if (chemin.contains("cat") || chemin.contains("chat")) {
			return 0; // Chat
		} else if (chemin.contains("dog") || chemin.contains("chien")) {
			return 1; // Chien
		} else if (chemin.contains("wild") || chemin.contains("sauvage")) {
			return 2; // Wild
		}
		return 3; // Inconnu par sécurité
	}

	private static double[] normaliserTableau(int[] donneesBrutes) {
		double[] donneesNormalisees = new double[donneesBrutes.length];
		for (int i = 0; i < donneesBrutes.length; i++) {
			donneesNormalisees[i] = donneesBrutes[i] / 255.0;
		}
		return donneesNormalisees;
	}

	private static double[] convertirEnDouble(int[] donneesBrutes) {
		double[] donneesDouble = new double[donneesBrutes.length];
		for (int i = 0; i < donneesBrutes.length; i++) {
			donneesDouble[i] = (double) donneesBrutes[i];
		}
		return donneesDouble;
	}

	private static double[] appliquerBruit(double[] source, double niveauBruit, boolean estNormalise) {
		double[] bruite = new double[source.length];
		Random rand = new Random();

		double borneMax = estNormalise ? 1.0 : 255.0;
		double amplitudeBruit = estNormalise ? niveauBruit : niveauBruit * 255.0;

		for (int i = 0; i < source.length; i++) {
			double perturbation = (rand.nextDouble() * 2.0 - 1.0) * amplitudeBruit;
			bruite[i] = Math.max(0.0, Math.min(borneMax, source[i] + perturbation));
		}
		return bruite;
	}
}