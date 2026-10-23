# SpiderVision

A Spider-Man-inspired camera HUD built with Python 3.12, OpenGL, and MediaPipe. It combines hand-tracked visual web effects, offline voice commands, an educational anatomy atlas, and an opt-in approximate-area map. **This is an interactive visual prototype—not a security system, medical device, threat detector, or physical web shooter.**

## Run the app

From the `GESTURE_SPIDERMAN` directory:

```bash
bash run_app.sh
```

The launcher creates `SpiderVision/.venv` and installs missing dependencies. Allow **camera and microphone** access on macOS. Voice recognition uses the local English Vosk model in `SpiderVision/assets/models/vosk-model-small-en-us-0.15/`; it does not send microphone recordings to a speech service. If the microphone is unavailable, press **M** for the menu and **1–9** to select an option.

## First-time setup and face session

1. Press **E** to enroll your face using eight camera samples, and stay in front of the webcam for a few seconds. The local face template is saved to `SpiderVision/.face_identity.npz` with owner-only file permissions. Older `.face_identity.yml` enrollments cannot be reused after the SFace upgrade; enroll again with **E**.
2. Wait for **SCANNING** followed by **SCAN COMPLETE**. On success, the app says *“Welcome, Mister. Face scan complete. Spider Vision online.”* Verification lasts for the app session. Briefly leaving the camera view does not require another enrollment; if another person causes the session to lock, the owner must be verified again.
3. After several consecutive mismatched face frames, **DANGER - UNKNOWN FACE** appears and gesture/menu controls lock. If the owner and someone else are detected together, **DANGER - UNKNOWN PERSON** appears with *“Danger, Mister. Unknown person detected ahead.”* With no face in view, the status is **NO FACE**, not DANGER; an already active owner session remains active. **These are prototype UI cues, not reliable security alerts.**
4. Press **D** to delete the local face template, then **E** to enroll again.

Face detection and recognition use the local OpenCV Zoo YuNet and SFace models in `SpiderVision/assets/models/`. Their license texts, along with the bundled Vosk model's license and upstream attributions, are documented in [the model-license notes](SpiderVision/assets/models/README.md). Photos or other people might be accepted incorrectly, and the owner might be rejected. **Do not use this prototype for real authentication or safety decisions.** The local face and voice recognizers do not upload face templates or microphone recordings. On macOS, spoken replies use the built-in Daniel voice offline; on other platforms, `edge-tts` requires an internet connection.

## Controls

| Input | Action |
| --- | --- |
| Extend index finger, little finger, and thumb (Spider-Man pose), or pinch thumb and index finger | Hold briefly to shoot a visual web from the index fingertip, including downward. Both hands can shoot independently; release before shooting again. Shooting is disabled while the anatomy atlas is open. |
| **Jarvis menu** | Open the menu and hear a short response. |
| **Menu** / **Open menu** / **Close menu** | Open or close the menu without saying “Jarvis.” |
| **Jarvis audio / performance / effects / screenshot** | Select a feature by voice. |
| **Jarvis aim** / **Jarvis web aim** | Toggle Web Aim Mode: reticle, aim line, ready indicator, and shot flash. |
| **Jarvis war mode on** / **Jarvis war mode off** | Toggle the red visual-simulation HUD and larger webs; close the shooting hand to retract its war web. |
| **Jarvis body** / **Jarvis anatomy** / **Jarvis brain** | Open the educational body or brain illustration. |
| **Jarvis zoom in** / **Jarvis zoom out** | Zoom the anatomy atlas. |
| **Close** / **Jarvis close** / **Jarvis close anatomy** / **Jarvis close body** / **Jarvis close brain** | Close the open atlas; when no atlas is open, `Close` closes the menu. |
| **Jarvis describe** / **Jarvis tell me about this** | Read out the selected anatomy label. |
| **Jarvis status** / **Jarvis what can you do** | Hear the mode status or available commands. |
| **Jarvis map** / **Jarvis show map** | Show the upper-right map panel; without a location it reads `LOCATION OFF`. |
| **Jarvis locate me**, then **Jarvis allow location** | After an explanation and a second explicit confirmation, request an approximate area from the public IP address. |
| **Jarvis hide map** / **Jarvis clear location** | Hide the map and clear the active area from app memory. |
| **Jarvis help** / **Jarvis exit** | Hear help or exit the app. |
| **Jarvis enroll face** | Re-enroll a face after verification; use **E** for initial enrollment. |
| **M**, then **1–9** | Keyboard menu after verification; **6** Web Aim, **7** body, **8** brain, **9** exit. |
| **Backspace** | Close the anatomy atlas without exiting. |
| **S** | Save a screenshot with the HUD after face verification. |
| **Space** | Test a visual web shot after face verification. |
| **E** / **D** | Enroll / delete the local face template. |
| **Esc** / **Q** | Exit. |

Screenshots are saved in `SpiderVision/screenshots/`. The menu also controls audio, performance and particle overlays, Web Aim, the anatomy atlas, screenshots, and face enrollment.

### Web Aim and war mode

Web Aim is **off by default**. After face verification, say **“Jarvis aim”**, point your index finger at the camera, and use the shooting pose or pinch. The **AIM READY** reticle and shot flash follow the latest fingertip position. They estimate a visual web direction; **they do not lock onto or identify real targets**. Actual responsiveness depends on camera and computer frame rate.

War mode is also **off by default** and requires a verified face session with the anatomy atlas closed. Say **“Jarvis war mode on”** to hear a reply and display the red **WAR MODE - VISUAL SIMULATION** HUD. New war webs look wider and remain visible for up to approximately **8 seconds**. Shoot, then close **the same hand** to retract its web. Fold the index finger and most other fingers; the thumb may remain slightly open. Retraction reads the latest hand landmarks rather than waiting for the slower smoothed hand pose. On a successful pull, a moving ring and **WEB RETRACTING** label, sparks, and a locally generated sound play for an approximately **0.8-second** return animation. A new war shot replaces the previous war web from that hand; the other hand is independent. Say **“Jarvis war mode off”** to leave the mode; face-session locking also turns it off. **This is only an on-screen effect, not a real danger warning, attack, or physical pull.**

### Educational anatomy atlas

**BODY** and **BRAIN** show detailed hologram-style illustrations of a full human figure and a lateral brain. Labels and connector lines identify parts, with a short explanation for a selection. Point at a hotspot for about 0.65 seconds or pinch; say **“Jarvis describe”** to hear its explanation. Move two index fingertips apart or together to zoom, or use the voice commands. Close with **“Close”**, **“Jarvis close anatomy”**, or **Backspace**. Art credits and licenses are in `SpiderVision/assets/anatomy/ATTRIBUTION.md`. **These are shared educational illustrations, not a scan of the camera user, a medical examination, or a diagnosis. A normal webcam cannot see internal organs or brain conditions.** For general terminology, see [MedlinePlus Anatomy](https://medlineplus.gov/anatomy.html).

### Approximate-area map and privacy

The small upper-right map can show only an **approximate area**, not GPS coordinates, a home address, a birthplace, or the location of someone seen by the camera. By default it says `LOCATION OFF` and makes no location guess at startup. To request a network-based area, say **“Jarvis locate me”** and listen to the notice that [ipapi.co](https://ipapi.co/api/) will receive your device's public IP address. Say **“Jarvis allow location”** for the separate confirmation that starts the lookup. The app keeps rounded coordinates (about 0.1° precision) only for the session; VPNs and network providers can shift the result. Map tiles then come from [OpenStreetMap](https://www.openstreetmap.org/) with contributor attribution. Without consent or internet access, it does not fabricate a location.

To avoid an IP-based lookup, supply an area yourself when launching from the project root, for example `SPIDERVISION_MAP_COORDS="LAT,LON" bash run_app.sh`, then say **“Jarvis map”**. Replace `LAT,LON` with coordinates **you choose**; this does not detect your device. Tiles are cached at `~/.cache/spidervision/osm-tiles/` with owner-only permissions. Tile filenames can reveal the approximate area even after you hide the map. Precise device location is not available from this launcher; macOS CoreLocation permissions have not been integrated and verified.

### Voice and simulated status panels

Speak one clear English command, then pause briefly for recognition. Except for **Menu** and **Close**, commands require **Jarvis**; **Jervis** is also accepted. The upper-right status shows **MIC LOADING**, **MIC READY**, **HEARD MENU**, **COMMAND NOT RECOGNIZED**, or **MIC OFF - PRESS M**. Wait while **JARVIS SPEAKING - WAIT** is displayed. Before face verification, **VERIFY FACE FIRST** appears and the menu remains locked. Jarvis supports **specific app commands and short descriptions**, not open-ended AI conversation.

After face verification, the left-middle panel shows animated **HEARTBEAT** and **BREATHING** effects. Its **PULSE FX / BREATH FX - SIMULATED** labels mean these are **not measurements of heart rate or breathing**; do not use them for health decisions. The HUD scales to the laptop display. If a panel does not appear, verify the face session and restart the app to load the current version. Muting audio says “Audio off” before disabling speech; turn it back on by voice or menu.

## Verify Jarvis voice recognition with your microphone

Close SpiderVision first so the app and verifier do not compete for the microphone. From the project root:

```bash
cd SpiderVision
.venv/bin/python scripts/verify_jarvis_voice.py --probe
.venv/bin/python scripts/verify_jarvis_voice.py
```

`--probe` checks that the model and microphone can open; **it does not show that your speech is understood**. In the guided run, press Enter and say each English phrase (including **Jarvis war mode on/off**); it reports `PASS`, a different recognized phrase, or `TIMEOUT`. Add `--timeout 20` if you need longer. On macOS, check Terminal's microphone permission if access is denied. This local Vosk check **does not save audio, dispatch app actions, or contact a location service**, even when testing “Jarvis allow location.” A pass shows that the microphone, recognizer, and command parser worked for that phrase in that session; test Jarvis replies and visuals separately in the running app. This is neither voice-identity verification nor general conversation.

## Run tests

From the project root:

```bash
cd SpiderVision
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

## License

The original SpiderVision source code is available under the [MIT License](LICENSE). Bundled third-party models retain their own terms: see the [model licenses and attribution](SpiderVision/assets/models/README.md). The anatomy artwork also has separate credits and licenses in [its attribution file](SpiderVision/assets/anatomy/ATTRIBUTION.md). The MIT license does not replace those third-party terms. Private face-enrollment data, screenshots, and local environments are not part of the public source release.

## Limitations

Gesture and face-recognition accuracy varies with lighting, pose, and camera quality. The virtual 1920×1080 HUD canvas is fitted to the laptop display; visual quality and speed depend on hardware. Speech output through `edge-tts` on non-macOS systems needs internet access. Face recognition is not real security, the anatomy atlas is not a medical tool, and the pulse/breathing display is not a sensor reading.
