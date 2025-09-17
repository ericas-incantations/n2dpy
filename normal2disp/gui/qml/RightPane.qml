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
        radius: theme.cornerRadius
        border.color: theme.border
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: theme.padding
        spacing: theme.spacing

        Label {
            text: "Normal Preview"
            color: theme.textPrimary
            font.pixelSize: 18
            Layout.alignment: Qt.AlignLeft
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            color: theme.textSecondary
            text: !backend || backend.inspectSummary === ""
                  ? "Inspect a mesh to populate materials, UV sets, and UDIMs."
                  : backend.inspectSummary
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            visible: backend && backend.warningSummary !== ""
            color: theme.accent
            text: backend ? backend.warningSummary : ""
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: theme.surfaceAlt
            radius: theme.cornerRadius
            border.color: theme.border

            Label {
                anchors.centerIn: parent
                text: "Normal map preview"
                color: theme.textMuted
            }
        }

        Expander {
            visible: backend && backend.udimTileCount > 1
            Layout.fillWidth: true
            Layout.preferredHeight: implicitHeight
            text: "Advanced"

            contentItem: ColumnLayout {
                anchors.fill: parent
                spacing: theme.spacing / 2

                Label {
                    text: "UDIM Tiles"
                    color: theme.textPrimary
                }

                Repeater {
                    model: backend ? backend.udimTiles : []

                    delegate: Label {
                        text: modelData
                        color: theme.textSecondary
                    }
                }
            }
        }
    }
}
