import java.io.*;
import java.util.*;
import javax.imageio.*;
import java.awt.image.*;
import java.nio.file.*;
import java.util.stream.*;

public class Image
{
	static private int LabelChat = 0;
	static private int LabelChien = 1;
	static private int LabelWild = 2;
	static private int LabelInconnu = 3;
	private int label = -1;
	private int largeur = 0;
	private int hauteur = 0;
	private int[] donnees = null; // image applatie en concaténant les lignes les unes après les autres

	public int label() {return label;}
	public int largeur() {return largeur;}
	public int hauteur() {return hauteur;}
	public int taille() {return donnees.length;} // nombre de pixels: hauteur*largeur ou 3*hauteur*largeur pour une image RGB
	public int[] donnees() {return donnees;}

	public boolean estEnNiveauxDeGris() {return taille() == largeur() * hauteur();}

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
			final int taille = niveauxDeGris ? hauteur*largeur : 3*hauteur*largeur;
			donnees = new int[taille];
			for (int i = 0; i < hauteur; ++i) {
				for (int j = 0; j < largeur; ++j) {
					final long rgb = img.getRGB(j, i);
					final int r = (int)((rgb>>16)&255);	// Isoler la composante rouge
					final int g = (int)((rgb>>8)&255);	// Isoler la composante verte
					final int b = (int)((rgb)&255);		// Isoler la composante bleue
					final int index = i * largeur + j;
					if (niveauxDeGris) {
						final float gris = 0.2125f * r + 0.7154f * g + 0.0721f * b; // RGB -> niveaux de gris
						donnees[index] = (int) Math.max(0, Math.min(255, gris));
					}
					else {
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

	public static List<String> listeFichiers(String repertoire) {
		List<String> cheminsFichiers = null;
		try {
			// La syntaxe qui suit enchaîne plusieurs méthodes d'affilée
			cheminsFichiers = Files.walk(Paths.get(repertoire))	// Récupère les chemins
				.filter(Files::isRegularFile)					// filtre uniquement les fichiers
				.map(Path::toAbsolutePath)						// convertit le chemin en chemin absolu
				.map(Path::toString)							// convertit le chemin en chaine de caractères
				.collect(Collectors.toList());					// crée une collection à partir de ces chaînes
		} catch (Exception e) {
			e.printStackTrace();
		}
		return cheminsFichiers;
	}

	public static void main (String[] args)
	{
		List<String> cheminsFichiers = listeFichiers("dataset_animaux/");
		for (String chemin : cheminsFichiers) {
			System.out.println(chemin);
		}

		final String chemin = "dataset_animaux/train/dog/010552.jpg";
		final int labelImage = chemin.indexOf("dog") != -1 ? LabelChien : LabelInconnu;
		Image im1 = new Image(chemin, labelImage, false);
		Image im2 = new Image(chemin, labelImage, true);
		im1.afficheMetadonnees();
		im2.afficheMetadonnees();
	}
}
