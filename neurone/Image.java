
import java.io.*;
import java.util.*;
import javax.imageio.*;
import java.awt.image.*;
import java.nio.file.*;
import java.util.stream.*;

public class Image {
	static private int LabelChat = 0;
	static private int LabelChien = 1;
	static private int LabelWild = 2;
	static private int LabelInconnu = 3;
	private int label = -1;
	private int largeur = 0;
	private int hauteur = 0;
	private int[] donnees = null; // image applatie en concaténant les lignes les unes après les autres

	public int label() {
		return label;
	}

	public int largeur() {
		return largeur;
	}

	public int hauteur() {
		return hauteur;
	}

	public int taille() {
		return donnees.length;
	} // nombre de pixels: hauteur*largeur ou 3*hauteur*largeur pour une image RGB

	public int[] donnees() {
		return donnees;
	}

	public boolean estEnNiveauxDeGris() {
		return taille() == largeur() * hauteur();
	}

	public void afficheMetadonnees() {
		String type = estEnNiveauxDeGris() ? "grayscale" : " couleurs";
		System.out.printf("Image (%s): label=%d, largeur=%d, hauteur=%d, taille=%d\n",
				type, label(), largeur(), hauteur(), taille());
	}

	public Image(final String cheminImage, int label, boolean niveauxDeGris) {
		try {
			final BufferedImage img = ImageIO.read(new File(cheminImage));
			this.label = label;
			largeur = img.getWidth(null);
			hauteur = img.getHeight(null);
			final int taille = niveauxDeGris ? hauteur * largeur : 3 * hauteur * largeur;
			donnees = new int[taille];
			for (int i = 0; i < hauteur; ++i) {
				for (int j = 0; j < largeur; ++j) {
					final long rgb = img.getRGB(j, i);
					final int r = (int) ((rgb >> 16) & 255); // Isoler la composante rouge
					final int g = (int) ((rgb >> 8) & 255); // Isoler la composante verte
					final int b = (int) ((rgb) & 255); // Isoler la composante bleue
					final int index = i * largeur + j;
					if (niveauxDeGris) {
						final float gris = 0.2125f * r + 0.7154f * g + 0.0721f * b; // RGB -> niveaux de gris
						donnees[index] = (int) Math.max(0, Math.min(255, gris));
					} else {
						donnees[3 * index + 0] = r;
						donnees[3 * index + 1] = g;
						donnees[3 * index + 2] = b;
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.err.printf("Image non trouvée ou non lisible: %s\n", cheminImage);
		}
	}

	public static List<String> listeFichiers(String repertoire) {
		List<String> cheminsFichiers = null;
		try {
			// La syntaxe qui suit enchaîne plusieurs méthodes d'affilée
			cheminsFichiers = Files.walk(Paths.get(repertoire)) // Récupère les chemins
					.filter(Files::isRegularFile) // filtre uniquement les fichiers
					.map(Path::toAbsolutePath) // convertit le chemin en chemin absolu
					.map(Path::toString) // convertit le chemin en chaine de caractères
					.collect(Collectors.toList()); // crée une collection à partir de ces chaînes
		} catch (Exception e) {
			e.printStackTrace();
		}
		return cheminsFichiers;
	}

	// =========================================================================
	// DEBUT DU CODE AJOUTÉ (Nouvelles fonctionnalités demandées)
	// =========================================================================

	/**
	 * Détermine automatiquement le label d'une image en fonction du nom du dossier
	 * qui la contient (cat/chat, dog/chien, wild).
	 */
	public static int determinerLabelDepuisChemin(String cheminFichier) {
		String cheminMinuscule = cheminFichier.toLowerCase();
		if (cheminMinuscule.contains("/dog/") || cheminMinuscule.contains("/chien/")) {
			return LabelChien;
		} else if (cheminMinuscule.contains("/cat/") || cheminMinuscule.contains("/chat/")) {
			return LabelChat;
		} else if (cheminMinuscule.contains("/wild/")) {
			return LabelWild;
		}
		return LabelInconnu;
	}

	/**
	 * Normalise les amplitudes des pixels du tableau 'donnees' [0, 255]
	 * vers un nouveau tableau de doubles [0.0, 1.0].
	 */
	public double[] normaliserAmplitudes() {
		if (this.donnees == null) {
			return null;
		}
		double[] donneesNormalisees = new double[this.donnees.length];
		for (int i = 0; i < this.donnees.length; i++) {
			donneesNormalisees[i] = this.donnees[i] / 255.0;
		}
		return donneesNormalisees;
	}

	/**
	 * Mélange de manière aléatoire la liste des données d'apprentissage (les
	 * images).
	 */
	public static void melangerDataset(List<Image> dataset) {
		if (dataset != null) {
			Collections.shuffle(dataset);
		}
	}

	/**
	 * Méthode utilitaire pour charger toutes les images d'un répertoire d'un coup,
	 * en leur attribuant automatiquement leur label.
	 */
	public static List<Image> chargerDatasetComplet(String repertoire, boolean niveauxDeGris) {
		List<String> chemins = listeFichiers(repertoire);
		List<Image> dataset = new ArrayList<>();

		if (chemins != null) {
			for (String chemin : chemins) {
				int labelAutomatique = determinerLabelDepuisChemin(chemin);
				dataset.add(new Image(chemin, labelAutomatique, niveauxDeGris));
			}
		}
		return dataset;
	}

	public static void main(String[] args) {
		// 1. Correction du chemin pour pointer vers TON dossier réel
		List<String> cheminsFichiers = listeFichiers("dataset_groupe_6/train/");

		if (cheminsFichiers == null || cheminsFichiers.isEmpty()) {
			System.out.println("Erreur : Aucun fichier trouvé dans le dossier 'dataset_groupe_6/train/' !");
			System.out.println("Vérifie que le dossier est bien placé à la racine de ton projet VS Code.");
			return;
		}

		System.out.println("--- ANALYSE ET TEST DES IMAGES DU DATASET ---");
		// Pour éviter d'inonder la console avec des milliers d'images,
		// on teste le chargement sur les 5 premières trouvées
		int limite = Math.min(5, cheminsFichiers.size());

		for (int i = 0; i < limite; i++) {
			String chemin = cheminsFichiers.get(i);

			// Détermination du label automatique selon le nom du dossier
			int labelAuto = 3; // Inconnu par défaut
			String cheminMin = chemin.toLowerCase();
			if (cheminMin.contains("cat") || cheminMin.contains("chat"))
				labelAuto = 0;
			else if (cheminMin.contains("dog") || cheminMin.contains("chien"))
				labelAuto = 1;
			else if (cheminMin.contains("wild") || cheminMin.contains("sauvage"))
				labelAuto = 2;

			// Chargement de l'image (en niveaux de gris pour le test)
			Image img = new Image(chemin, labelAuto, true);

			// Si l'image a pu être lue correctement, on affiche ses métadonnées
			if (img.donnees() != null) {
				img.afficheMetadonnees();
			} else {
				System.err.printf("Échec critique du chargement pour : %s\n", chemin);
			}
		}
	}
}