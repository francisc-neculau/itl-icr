
class Paths:

    @staticmethod
    def project_directory():
        return "C:\ComputerScience\source\itl-icr\org\itl\icr\\"

    @staticmethod
    def models():
        return Paths.project_directory() + "isr\\ai\\models\\"

    @staticmethod
    def nn_model():
        return Paths.models() + "nn\\"

    @staticmethod
    def resources():
        return Paths.project_directory() + "resources\\"

    @staticmethod
    def equations():
        return Paths.resources() + "equation_images\\"

    @staticmethod
    def dataset_inftycdb3():
        return Paths.resources() + "dataset\\InftyCDB-3\\"

    @staticmethod
    def character_palette():
        return Paths.resources() + "characterPalette.jpg"

    @staticmethod
    def external_resources():
        return "C:\\ComputerScience\\resources\\[Processed] InftyCDB-3_new"


