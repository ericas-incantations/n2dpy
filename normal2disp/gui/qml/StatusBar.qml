import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root
    property var theme
    property var backend

    Rectangle {
        anchors.fill: parent
        color: theme.surface
        border.color: theme.border
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: theme.padding
        spacing: theme.spacing

        RowLayout {
            Layout.fillWidth: true
            spacing: theme.spacing

            ColumnLayout {

                Layout.fillWidth: true
                spacing: theme.spacing / 4

                Label {
                    text: backend ? backend.statusMessage : "Ready"
                    color: theme.textPrimary
                    font.pixelSize: 14
                    Layout.fillWidth: true
                }

                Label {
                    text: backend ? backend.progressDetail : ""
                    visible: backend && backend.progressDetail !== ""
                    color: theme.textSecondary
                    font.pixelSize: 12
                    Layout.fillWidth: true
                }

                Label {
                    text: backend ? backend.displacementAmplitude : ""
                    visible: backend && backend.displacementAmplitude !== ""
                    color: theme.textSecondary
                    font.pixelSize: 12
                    Layout.fillWidth: true
                }
            }

            ProgressBar {
                indeterminate: backend && backend.inspectRunning && !(backend.bakeRunning)
                value: backend ? backend.progressValue : 0

                from: 0
                to: 1
                Layout.preferredWidth: 220
            }

            Button {
                text: "Cancel"
                enabled: backend && backend.bakeRunning
                onClicked: if (backend) backend.cancelBake()

            }

            Button {
                text: "Open Output"
                enabled: backend && backend.canOpenOutput
                onClicked: if (backend) backend.openOutputFolder()
            }

            Button {
                text: "Reveal EXR"
                enabled: backend && backend.canRevealLatestOutput
                onClicked: if (backend) backend.revealLatestOutput()
            }
        }

        Frame {
            Layout.fillWidth: true
            Layout.fillHeight: true
            background: Rectangle {
                color: theme.surfaceAlt
                border.color: theme.border
            }

            ScrollView {
                anchors.fill: parent
                TextArea {
                    text: backend ? backend.logText : "Waiting for job output..."
                    readOnly: true
                    color: theme.textSecondary
                    background: null
                    wrapMode: Text.Wrap
                }
            }
        }
    }
}
