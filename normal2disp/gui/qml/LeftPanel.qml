import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3

Item {
    id: root
    property var theme

    property string meshPath: ""
    property string normalPath: ""
    property string outputPath: ""

    signal meshSelected(string path)
    signal normalSelected(string path)
    signal outputSelected(string path)

    function cleanPath(url) {
        if (!url) {
            return ""
        }
        var stringValue = url.toString()
        if (stringValue.startsWith("file:///")) {
            stringValue = stringValue.substring(8)
        } else if (stringValue.startsWith("file://")) {
            stringValue = stringValue.substring(7)
        }
        return decodeURIComponent(stringValue)
    }

    Rectangle {
        anchors.fill: parent
        color: theme.surface
        radius: theme.cornerRadius
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: theme.padding
        clip: true

        ColumnLayout {
            id: content
            spacing: theme.spacing
            width: parent.width

            Label {
                text: "Project"
                color: theme.textSecondary
                font.pixelSize: 18
                Layout.alignment: Qt.AlignLeft
            }

            Frame {
                Layout.fillWidth: true
                background: Rectangle {
                    color: theme.surfaceAlt
                    radius: theme.cornerRadius
                    border.color: theme.border
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: theme.padding
                    spacing: theme.spacing

                    Label {
                        text: "Mesh"
                        color: theme.textPrimary
                    }

                    RowLayout {
                        spacing: theme.spacing
                        Layout.fillWidth: true

                        Button {
                            text: "Browse"
                            Layout.preferredWidth: 110
                            onClicked: meshDialog.open()
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WrapAnywhere
                            color: theme.textSecondary
                            text: meshPath === "" ? "No mesh selected" : meshPath
                        }
                    }

                    Label {
                        text: "Normal map"
                        color: theme.textPrimary
                    }

                    RowLayout {
                        spacing: theme.spacing
                        Layout.fillWidth: true

                        Button {
                            text: "Browse"
                            Layout.preferredWidth: 110
                            onClicked: normalDialog.open()
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WrapAnywhere
                            color: theme.textSecondary
                            text: normalPath === "" ? "No normal map selected" : normalPath
                        }
                    }

                    Label {
                        text: "Output directory"
                        color: theme.textPrimary
                    }

                    RowLayout {
                        spacing: theme.spacing
                        Layout.fillWidth: true

                        Button {
                            text: "Browse"
                            Layout.preferredWidth: 110
                            onClicked: outputDialog.open()
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WrapAnywhere
                            color: theme.textSecondary
                            text: outputPath === "" ? "No output folder selected" : outputPath
                        }
                    }
                }
            }

            Label {
                text: "Bake options"
                color: theme.textSecondary
                font.pixelSize: 18
                Layout.alignment: Qt.AlignLeft
            }

            Frame {
                Layout.fillWidth: true
                background: Rectangle {
                    color: theme.surfaceAlt
                    radius: theme.cornerRadius
                    border.color: theme.border
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: theme.padding
                    spacing: theme.spacing

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "UV Set"
                            color: theme.textPrimary
                        }

                        ComboBox {
                            Layout.fillWidth: true
                            model: ["Auto", "UVSet1", "UVSet2"]
                            enabled: false
                        }
                    }

                    CheckBox {
                        text: "Y is down"
                        enabled: false
                        checked: false
                        font.pixelSize: 14
                        Layout.alignment: Qt.AlignLeft
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "Amplitude"
                            color: theme.textPrimary
                        }

                        Slider {
                            id: amplitudeSlider
                            from: 0
                            to: 10
                            value: 1
                            Layout.fillWidth: true
                            onMoved: amplitudeSpin.value = value
                        }

                        SpinBox {
                            id: amplitudeSpin
                            from: 0
                            to: 10
                            stepSize: 0.1
                            value: amplitudeSlider.value
                            editable: true
                            Layout.preferredWidth: 80
                            onValueModified: amplitudeSlider.value = value
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "Max slope"
                            color: theme.textPrimary
                        }

                        Slider {
                            id: slopeSlider
                            from: 1
                            to: 60
                            value: 10
                            Layout.fillWidth: true
                            onMoved: slopeSpin.value = value
                        }

                        SpinBox {
                            id: slopeSpin
                            from: 1
                            to: 60
                            stepSize: 1
                            value: slopeSlider.value
                            editable: true
                            Layout.preferredWidth: 80
                            onValueModified: slopeSlider.value = value
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "Normalization"
                            color: theme.textPrimary
                        }

                        ComboBox {
                            Layout.fillWidth: true
                            model: ["Auto", "XYZ", "XY", "None"]
                            currentIndex: 0
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: theme.spacing

                        Label {
                            text: "Processes"
                            color: theme.textPrimary
                        }

                        SpinBox {
                            from: 0
                            to: 32
                            value: 0
                            Layout.preferredWidth: 80
                        }
                    }

                    CheckBox {
                        text: "Deterministic"
                        checked: false
                        Layout.alignment: Qt.AlignLeft
                    }

                    CheckBox {
                        text: "Export sidecars"
                        checked: false
                        Layout.alignment: Qt.AlignLeft
                    }

                    Button {
                        text: "Inspect Mesh"
                        enabled: false
                        Layout.fillWidth: true
                        onClicked: console.log("Inspect requested (placeholder)")
                    }

                    Button {
                        text: "Bake"
                        Layout.fillWidth: true
                        highlighted: true
                        onClicked: console.log("Bake requested (placeholder)")
                    }
                }
            }
        }
    }

    FileDialog {
        id: meshDialog
        title: "Select mesh"
        nameFilters: ["Meshes (*.fbx *.obj *.gltf *.glb)", "All files (*)"]
        onAccepted: {
            meshPath = cleanPath(selectedFile)
            meshSelected(meshPath)
        }
    }

    FileDialog {
        id: normalDialog
        title: "Select normal map"
        nameFilters: ["Images (*.png *.exr *.tif *.tiff)", "All files (*)"]
        onAccepted: {
            normalPath = cleanPath(selectedFile)
            normalSelected(normalPath)
        }
    }

    FolderDialog {
        id: outputDialog
        title: "Select output directory"
        onAccepted: {
            outputPath = cleanPath(selectedFolder)
            outputSelected(outputPath)
        }
    }
}
