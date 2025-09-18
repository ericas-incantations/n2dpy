import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick3D
import QtQuick3D.Helpers

Item {
    id: root
    property var theme
    property var backend
    property bool wireframe: false

    Rectangle {
        anchors.fill: parent
        radius: theme.cornerRadius
        color: theme.surface
        border.color: theme.border
    }

    View3D {
        id: view3d
        anchors.fill: parent
        anchors.margins: theme.padding
        environment: SceneEnvironment {
            clearColor: theme.surfaceAlt
            backgroundMode: SceneEnvironment.Color
        }
        function configureAntialiasing() {
            if ("antialiasingMode" in view3d) {
                view3d.antialiasingMode = View3D.MSAA
                if ("antialiasingQuality" in view3d) {
                    view3d.antialiasingQuality = View3D.AntialiasingQualityHigh
                }
            } else if ("multisampleAAMode" in view3d) {
                view3d.multisampleAAMode = View3D.MSAA4X
            }
        }

        Component.onCompleted: configureAntialiasing()

        Node {
            id: sceneRoot

            DirectionalLight {
                id: keyLight
                eulerRotation.x: backend ? backend.lightElevation : -35
                eulerRotation.y: backend ? backend.lightAzimuth : 35
                brightness: 80
            }

            Model {
                id: previewModel
                source: backend && backend.meshSource !== "" ? backend.meshSource : ""
                materials: backend && backend.displacementEnabled ? [displacementMaterial] : [previewMaterial]
                castsShadows: true
                visible: backend && (
                             backend.displacementEnabled
                                 ? backend.displacementGeometry === null
                                 : backend.meshSource !== ""
                         )
            }

            Model {
                id: displacementModel
                geometry: backend && backend.displacementGeometry ? backend.displacementGeometry : null
                materials: [displacementMaterial]
                castsShadows: true
                visible: backend && backend.displacementEnabled && backend.displacementGeometry
            }

            Model {
                id: placeholderModel
                source: "#Sphere"
                materials: [previewMaterial]
                scale: Qt.vector3d(0.6, 0.6, 0.6)
                visible: !backend || backend.meshSource === ""
            }
        }

        PerspectiveCamera {
            id: camera
            position: Qt.vector3d(0, 0, 6)
            clipNear: 0.01
            clipFar: 2500
        }

        OrbitCameraController {
            id: controller
            camera: camera
            target: sceneRoot
            linearSpeed: 2
            lookAt: Qt.vector3d(0, 0, 0)
            zoomSpeed: 0.2
            panSpeed: 0.8
        }
    }

    Texture {
        id: normalTexture
        source: backend ? backend.normalTextureUrl : ""
        colorSpace: Texture.Linear
    }

    PrincipledMaterial {
        id: previewMaterial
        baseColor: Qt.rgba(0.65, 0.67, 0.72, 1)
        roughness: 0.55
        metalness: 0.0
        wireframe: root.wireframe
        normalMap: backend && backend.normalTextureUrl !== "" && !(backend.displacementEnabled) ? normalTexture : null
        normalStrength: backend
                        ? (backend.displacementEnabled
                           ? 0.0
                           : (backend.normalEnabled ? 1.0 : 0.0))
                        : 0.0
    }

    PrincipledMaterial {
        id: displacementMaterial
        baseColor: Qt.rgba(0.65, 0.67, 0.72, 1)
        roughness: 0.55
        metalness: 0.0
        wireframe: root.wireframe
        normalStrength: 0.0
    }

    Rectangle {
        id: overlay
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: theme.padding
        radius: theme.cornerRadius
        color: theme.surface
        opacity: 0.9
        border.color: theme.border

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: theme.padding
            spacing: theme.spacing

            RowLayout {
                spacing: theme.spacing

                CheckBox {
                    id: wireframeToggle
                    text: "Wireframe"
                    checked: root.wireframe
                    onToggled: root.wireframe = checked
                }

                Item { Layout.fillWidth: true }

                CheckBox {
                    id: displacementToggle
                    text: "Displacement preview"
                    checked: backend && backend.displacementEnabled
                    enabled: backend && !backend.displacementBusy
                    onToggled: if (backend) backend.setDisplacementEnabled(checked)
                }
            }

            RowLayout {
                spacing: theme.spacing / 2
                visible: backend && (backend.displacementEnabled || backend.displacementBusy)

                Label {
                    text: "Subdivision"
                    color: backend && backend.displacementEnabled ? theme.textPrimary : theme.textSecondary
                }

                Slider {
                    id: subdivSlider
                    from: 0
                    to: 5
                    stepSize: 1
                    snapMode: Slider.SnapAlways
                    Layout.fillWidth: true
                    enabled: backend && backend.displacementEnabled && !backend.displacementBusy
                    onValueChanged: if (backend && pressed) backend.setDisplacementLevel(Math.round(value))
                }

                Binding {
                    target: subdivSlider
                    property: "value"
                    value: backend ? backend.displacementLevel : 0
                    when: !subdivSlider.pressed
                }

                Label {
                    text: backend ? backend.displacementLevel : 0
                    horizontalAlignment: Text.AlignHCenter
                    color: theme.textSecondary
                    Layout.preferredWidth: 32
                }

                Button {
                    id: regenerateButton
                    text: "Regenerate"
                    visible: backend && backend.displacementEnabled
                    enabled: backend && backend.displacementEnabled && backend.displacementDirty && !backend.displacementBusy
                    onClicked: if (backend) backend.regenerateDisplacement()
                }
            }

            RowLayout {
                spacing: theme.spacing / 2
                visible: backend && (backend.displacementEnabled || backend.displacementBusy)

                Label {
                    text: "Preview scale"
                    color: backend && backend.displacementEnabled ? theme.textPrimary : theme.textSecondary
                }

                SpinBox {
                    id: previewScale
                    from: -10
                    to: 10
                    stepSize: 0.1
                    editable: true
                    Layout.preferredWidth: 120
                    enabled: backend && backend.displacementEnabled && !backend.displacementBusy
                    onValueModified: if (backend) backend.setDisplacementPreviewScale(value)
                    validator: DoubleValidator {
                        bottom: -10
                        top: 10
                        decimals: 3
                    }
                    textFromValue: function(value, locale) { return Number(value).toFixed(2); }
                    valueFromText: function(text, locale) {
                        var parsed = parseFloat(text)
                        return isNaN(parsed) ? previewScale.value : parsed
                    }
                }

                Binding {
                    target: previewScale
                    property: "value"
                    value: backend ? backend.displacementPreviewScale : 1.0
                    when: !previewScale.activeFocus
                }
            }

            Label {
                text: backend ? backend.displacementStatus : ""
                visible: backend && backend.displacementStatus !== ""
                color: theme.textSecondary
                wrapMode: Text.WordWrap
                Layout.maximumWidth: 280
            }
        }
    }

    Column {
        anchors.centerIn: view3d
        spacing: theme.spacing / 2
        visible: backend && backend.displacementBusy

        BusyIndicator {
            running: true
            width: 48
            height: 48
        }

        Label {
            text: backend ? backend.displacementStatus : ""
            color: theme.textSecondary
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: 12
        }
    }
}
