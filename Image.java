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
	public static Object Mode;
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

	public static void main(String[] args) {
		List<String> cheminsFichiers = listeFichiers("dataset_groupe_6/");
		for (String chemin : cheminsFichiers) {
			System.out.println(chemin);
		}

		final String chemin = "dataset_groupe_6/train/dog/010552.jpg";
		final int labelImage = chemin.indexOf("dog") != -1 ? LabelChien : LabelInconnu;
		Image im1 = new Image(chemin, labelImage, false);
		Image im2 = new Image(chemin, labelImage, true);
		im1.afficheMetadonnees();
		im2.afficheMetadonnees();
	}

	public enum ModeCouleur {
		RGB, GRIS, TSL
	}

	/**
	 * Nouveau Constructeur optimisé
	 * Intègre : Labellisation automatique, Choix de couleur (RGB, Gris, TSL) et
	 * Effet Miroir
	 */
	public Image(final String cheminImage, ModeCouleur mode, boolean miroir) {
		try {
			// 1. LABELLISATION AUTOMATIQUE (grâce au nom du dossier dans le chemin)
			String cheminMinuscule = cheminImage.toLowerCase();
			if (cheminMinuscule.contains("/cat/") || cheminMinuscule.contains("\\cat\\")
					|| cheminMinuscule.contains("cat")) {
				this.label = LabelChat;
			} else if (cheminMinuscule.contains("/dog/") || cheminMinuscule.contains("\\dog\\")
					|| cheminMinuscule.contains("dog")) {
				this.label = LabelChien;
			} else if (cheminMinuscule.contains("/wild/") || cheminMinuscule.contains("\\wild\\")
					|| cheminMinuscule.contains("wild")) {
				this.label = LabelWild;
			} else {
				this.label = LabelInconnu;
			}

			final BufferedImage img = ImageIO.read(new File(cheminImage));
			this.largeur = img.getWidth(null);
			this.hauteur = img.getHeight(null);

			// Détermination de la taille du tableau selon le mode choisi
			final int canaux = (mode == ModeCouleur.GRIS) ? 1 : 3;
			this.donnees = new int[this.hauteur * this.largeur * canaux];

			for (int i = 0; i < this.hauteur; ++i) {
				for (int j = 0; j < this.largeur; ++j) {

					// 5. AUGMENTATION DE DONNÉES : Gestion du miroir horizontal
					// Si miroir est vrai, on inverse l'axe X (largeur - 1 - j)
					int xPixel = miroir ? (this.largeur - 1 - j) : j;

					final long rgb = img.getRGB(xPixel, i);
					final int r = (int) ((rgb >> 16) & 255);
					final int g = (int) ((rgb >> 8) & 255);
					final int b = (int) (rgb & 255);

					final int index = i * this.largeur + j;

					// 4. CHOIX DU TRAITEMENT COLORIMÉTRIQUE
					if (mode == ModeCouleur.GRIS) {
						final float gris = 0.2125f * r + 0.7154f * g + 0.0721f * b;
						this.donnees[index] = (int) Math.max(0, Math.min(255, gris));
					} else if (mode == ModeCouleur.RGB) {
						this.donnees[3 * index + 0] = r;
						this.donnees[3 * index + 1] = g;
						this.donnees[3 * index + 2] = b;
					} else if (mode == ModeCouleur.TSL) {
						// Conversion RGB -> TSL (Teinte, Saturation, Luminance)
						float r_f = r / 255.0f;
						float g_f = g / 255.0f;
						float b_f = b / 255.0f;

						float max = Math.max(r_f, Math.max(g_f, b_f));
						float min = Math.min(r_f, Math.min(g_f, b_f));

						float t = 0.0f, s = 0.0f, l = (max + min) / 2.0f;

						if (max != min) {
							float d = max - min;
							s = l > 0.5f ? d / (2.0f - max - min) : d / (max + min);
							if (max == r_f) {
								t = (g_f - b_f) / d + (g_f < b_f ? 6 : 0);
							} else if (max == g_f) {
								t = (b_f - r_f) / d + 2;
							} else if (max == b_f) {
								t = (r_f - g_f) / d + 4;
							}
							t /= 6.0f;
						}
						// Stockage temporaire quantifié entre 0 et 255 pour rester cohérent avec le
						// tableau int[]
						this.donnees[3 * index + 0] = (int) (t * 255);
						this.donnees[3 * index + 1] = (int) (s * 255);
						this.donnees[3 * index + 2] = (int) (l * 255);
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.err.printf("Erreur lors de l'initialisation avancée de l'image: %s\n", cheminImage);
		}
	}

	/**
	 * 2. NORMALISER LES AMPLITUDES DE PIXELS
	 * Convertit le tableau d'entiers [0, 255] en un tableau de float [0.0, 1.0]
	 * requis par le Neurone
	 */
	public float[] obtenirDonneesNormalisees() {
		if (this.donnees == null)
			return null;
		float[] donneesNormalisees = new float[this.donnees.length];
		for (int i = 0; i < this.donnees.length; i++) {
			donneesNormalisees[i] = this.donnees[i] / 255.0f;
		}
		return donneesNormalisees;
	}

	/**
	 * 3. MÉLANGER LES DONNÉES À APPRENDRE
	 * Implémentation de l'algorithme de mélange de Fisher-Yates
	 */
	public static void melangerDonnees(List<Image> listeImages) {
		if (listeImages == null || listeImages.size() <= 1)
			return;
		Random rand = new Random();
		for (int i = listeImages.size() - 1; i > 0; i--) {
			int indexAleatoire = rand.nextInt(i + 1);
			// Échange des objets Image
			Image temp = listeImages.get(indexAleatoire);
			listeImages.set(indexAleatoire, listeImages.get(i));
			listeImages.set(i, temp);
		}
	}
}
