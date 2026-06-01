import java.awt.image.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import javax.imageio.*;

public class Image
{
	// Labels rendus public pour être utilisables depuis ClassificationAnimaux
	static public int LabelChat    = 0;
	static public int LabelChien   = 1;
	static public int LabelWild    = 2;
	static public int LabelInconnu = 3;

	private int label   = -1;
	private int largeur = 0;
	private int hauteur = 0;
	private int[] donnees = null; // image applatie en concaténant les lignes les unes après les autres

	public int   label()   {return label;}
	public int   largeur() {return largeur;}
	public int   hauteur() {return hauteur;}
	public int   taille()  {return donnees == null ? 0 : donnees.length;}
	public int[] donnees() {return donnees;}

	public boolean estEnNiveauxDeGris() {return taille() == largeur() * hauteur();}

	public void afficheMetadonnees() {
		String type = estEnNiveauxDeGris() ? "grayscale" : "couleurs";
		System.out.printf("Image (%s): label=%d, largeur=%d, hauteur=%d, taille=%d\n",
			type, label(), largeur(), hauteur(), taille());
	}

	/** Constructeur depuis un fichier image */
	public Image(final String cheminImage, int label, boolean niveauxDeGris) {
		try {
			final BufferedImage img = ImageIO.read(new File(cheminImage));
			if (img == null) {
				System.err.printf("Image non lisible (format inconnu): %s\n", cheminImage);
				return;
			}
			this.label = label;
			largeur = img.getWidth(null);
			hauteur = img.getHeight(null);
			final int taille = niveauxDeGris ? hauteur*largeur : 3*hauteur*largeur;
			donnees = new int[taille];
			for (int i = 0; i < hauteur; ++i) {
				for (int j = 0; j < largeur; ++j) {
					final long rgb = img.getRGB(j, i);
					final int r = (int)((rgb>>16)&255);
					final int g = (int)((rgb>>8)&255);
					final int b = (int)((rgb)&255);
					final int index = i * largeur + j;
					if (niveauxDeGris) {
						final float gris = 0.2125f * r + 0.7154f * g + 0.0721f * b;
						donnees[index] = (int) Math.max(0, Math.min(255, gris));
					} else {
						donnees[3*index+0] = r;
						donnees[3*index+1] = g;
						donnees[3*index+2] = b;
					}
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			System.err.printf("Image non trouvée ou non lisible: %s\n", cheminImage);
		}
	}

	/**
	 * Constructeur interne utilisé par ClassificationAnimaux.souséchantillonner()
	 * pour créer une image déjà traitée (données fournies directement).
	 */
	public Image(int label, int largeur, int hauteur, int[] donnees) {
		this.label   = label;
		this.largeur = largeur;
		this.hauteur = hauteur;
		this.donnees = donnees;
	}

	/** Liste récursivement tous les fichiers d'un répertoire */
	public static List<String> listeFichiers(String repertoire) {
		List<String> cheminsFichiers = null;
		try {
			cheminsFichiers = Files.walk(Paths.get(repertoire))
				.filter(Files::isRegularFile)
				.map(Path::toAbsolutePath)
				.map(Path::toString)
				.collect(Collectors.toList());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return cheminsFichiers;
	}

	/** Main de test — affiche les métadonnées d'une image */
	public static void main(String[] args)
	{
		// Test rapide sur une seule image
		final String chemin = "dataset_groupe_6/train/dog/010552.jpg";
		final int labelImage = chemin.contains("dog") ? LabelChien : LabelInconnu;
		Image im1 = new Image(chemin, labelImage, false);
		Image im2 = new Image(chemin, labelImage, true);
		im1.afficheMetadonnees();
		im2.afficheMetadonnees();
	}
}