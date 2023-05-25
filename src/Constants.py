from models.encoder import CNN2DShapesEncoder, CNN3DShapesEncoder
from models.decoder import CNN2DShapesDecoder, CNN3DShapesDecoder

DATA_CLASSES = {
    'dsprites': (CNN2DShapesEncoder, CNN2DShapesDecoder),
    'shapes3d': (CNN3DShapesEncoder, CNN3DShapesDecoder),
    'car': (CNN3DShapesEncoder, CNN3DShapesDecoder),
}