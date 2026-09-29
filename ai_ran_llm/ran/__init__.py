"""Real-RAN integration for HandoverLLM (OCUDU / srsRAN, OpenAirInterface, any E2 node).

Layers (see docs/DESIGN.md §16.6 and docs/RAN_INTEGRATION.md):

* ``messages``   canonical messages + the ran-bridge NDJSON protocol
* ``rrc``        3GPP RRC MeasurementReport parsing (JSON / XER), TS 38.133 mappings
* ``cells``      RAN cell identities <-> model cell indices, neighbour relations
* ``tracker``    per-UE measurement history resampled to the model's input
* ``controller`` model decisions + guard rails -> handover commands
* ``actuators``  E2SM-RC (O-RAN SC RIC), OAI telnet, srsRAN/OCUDU console, command, bridge
* ``sources``    RRC log tailing
* ``bridge``     TCP transport for the ran-bridge protocol
* ``runtime``    the xApp event loop with audit log
* ``fake_gnb``   simulator-backed gNB for end-to-end tests
"""

from .cells import CellMap
from .controller import ControllerConfig, HandoverController
from .messages import CellMeas, CellRef, HandoverCommand, HandoverOutcome, MeasReport
from .tracker import UEMeasurementTracker

__all__ = ["CellMap", "ControllerConfig", "HandoverController", "CellMeas", "CellRef", "HandoverCommand",
           "HandoverOutcome", "MeasReport", "UEMeasurementTracker"]
