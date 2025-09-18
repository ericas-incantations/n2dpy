import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick3D 1.15
import QtQuick3D.Helpers 1.15

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
        multisampleAAMode: View3D.MSAA4X

        Node {
            id: sceneRoot

            DirectionalLight {
                id: keyLight
                eulerRotation.x: -35
                eulerRotation.y: 35
                brightness: 80
            }

            Model {
                id: previewModel
                source: backend && backend.meshSource !== "" ? backend.meshSource : ""
                materials: [previewMaterial]
                castsShadows: true
                visible: backend && backend.meshSource !== ""
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
        normalMap: backend && backend.normalTextureUrl !== "" ? normalTexture : null
        normalStrength: backend && backend.normalEnabled ? 1.0 : 0.0
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

        RowLayout {
            anchors.fill: parent
            anchors.margins: theme.padding
            spacing: theme.spacing

            CheckBox {
                id: wireframeToggle
                text: "Wireframe"
                checked: root.wireframe
                onToggled: root.wireframe = checked
            }
        }
    }
}
